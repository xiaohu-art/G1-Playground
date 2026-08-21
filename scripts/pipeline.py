# ruff: noqa: I001

import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time
from types import SimpleNamespace

# Run the Jetson torch-before-numpy bootstrap before anything that can import NumPy.
import g1_playground  # noqa: F401

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g1_playground.policy import UnitreeWoGaitPolicy
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import is_upright

logger = logging.getLogger("g1_playground")
RAMP_SECONDS = 3.0
BLEND_SECONDS = 5.0
LOG_CAPACITY = 50 * 60 * 20
ZERO_CONTROL = {"axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0}}


def read_frame(env, controller):
    state = env.read()
    control, shutdown_requested = controller.read()
    if shutdown_requested:
        logger.warning("Emergency shutdown!")
        return None
    if not is_upright(state.base_quat):
        logger.error("Robot fallen! Shutting down for safety.")
        return None
    return state, control


def step(env, controller, policy, *, send_command: bool = True) -> bool:
    frame = read_frame(env, controller)
    if frame is None:
        return False
    state, control = frame
    target = policy.act(state, control)
    if send_command:
        env.step(target)
    return True


def recorder(capacity: int):
    return SimpleNamespace(
        count=0,
        elapsed=np.zeros(capacity),
        dof_pos=np.zeros((capacity, 29), dtype=np.float32),
        dof_vel=np.zeros((capacity, 29), dtype=np.float32),
        base_quat=np.zeros((capacity, 4), dtype=np.float32),
        base_ang_vel=np.zeros((capacity, 3), dtype=np.float32),
        command=np.zeros((capacity, 29), dtype=np.float32),
        base_pos=np.full((capacity, 3), np.nan, dtype=np.float32),
        base_lin_vel=np.full((capacity, 3), np.nan, dtype=np.float32),
        body_height=np.full(capacity, np.nan, dtype=np.float32),
    )


def record(log, elapsed, state, command, odometry) -> None:
    index = log.count
    if index >= log.elapsed.shape[0]:
        return
    log.elapsed[index] = elapsed
    log.dof_pos[index] = state.dof_pos
    log.dof_vel[index] = state.dof_vel
    log.base_quat[index] = state.base_quat
    log.base_ang_vel[index] = state.base_ang_vel
    log.command[index] = command
    if odometry is not None:
        log.base_pos[index] = odometry.position
        log.base_lin_vel[index] = odometry.velocity
        log.body_height[index] = odometry.body_height
    log.count = index + 1


def save_recording(log, cfg) -> None:
    if log.count == 0:
        return
    directory = os.path.join("logs", time.strftime("state_%Y%m%d-%H%M%S"))
    os.makedirs(directory, exist_ok=True)
    fields = {name: value[: log.count] for name, value in vars(log).items() if name != "count"}
    np.savez_compressed(os.path.join(directory, "state.npz"), **fields)
    with open(os.path.join(directory, "config.yaml"), "w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    odometry = float(np.isfinite(log.base_pos[: log.count, 0]).mean())
    height = float(np.isfinite(log.body_height[: log.count]).mean())
    logger.warning(
        "Wrote %d frames to %s (odometry valid %.1f%%, body height valid %.1f%%)",
        log.count,
        directory,
        100.0 * odometry,
        100.0 * height,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="run_pipeline")
def run(cfg: DictConfig) -> None:
    setup_logger()
    env = None
    log = recorder(LOG_CAPACITY)
    try:
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        policy = UnitreeWoGaitPolicy(cfg.policy, device=cfg.device, dof_cfg=dof)
        env = instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)
        controller = instantiate(cfg.controller, env=env)

        env.self_check()
        for _ in range(10):
            if not step(env, controller, policy, send_command=False):
                return

        frame = read_frame(env, controller)
        if frame is None:
            return
        state, _ = frame
        initial = state.dof_pos
        standing = policy.standing_target
        env.activate_commands()

        logger.warning("Ramping to the policy standing pose over %.1f seconds", RAMP_SECONDS)
        ramp_steps = int(RAMP_SECONDS * policy.freq)
        for index in range(ramp_steps):
            started = time.monotonic()
            if read_frame(env, controller) is None:
                return
            alpha = (index + 1) / ramp_steps
            env.step((1.0 - alpha) * initial + alpha * standing)
            remaining = policy.dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -policy.dt:
                logger.warning("Control frame dropped during standing ramp")

        policy.reset()

        logger.warning("Blending into closed-loop locomotion over %.1f seconds", BLEND_SECONDS)
        blend_steps = int(BLEND_SECONDS * policy.freq)
        for index in range(blend_steps):
            started = time.monotonic()
            frame = read_frame(env, controller)
            if frame is None:
                return
            state, _ = frame
            policy_target = policy.act(state, ZERO_CONTROL)
            alpha = (index + 1) / blend_steps
            env.step((1.0 - alpha) * standing + alpha * policy_target)
            remaining = policy.dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -policy.dt:
                logger.warning("Control frame dropped during policy blend")

        origin = time.monotonic()
        while True:
            started = time.monotonic()
            frame = read_frame(env, controller)
            if frame is None:
                break
            state, control = frame
            target = policy.act(state, control)
            env.step(target)
            record(log, started - origin, state, target, env.read_odometry())
            remaining = policy.dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            else:
                logger.error("Warning: frame drop -> %s", remaining)
                if remaining < -0.2:
                    logger.critical("Exiting due to excessive frame drop")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        save_recording(log, cfg)
        if env is not None:
            env.shutdown()


if __name__ == "__main__":
    run()
