# ruff: noqa: I001

import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import logging
import time

# Apply the Jetson process settings before importing the numerical stack.
import g1_playground  # noqa: F401

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from g1_playground.policy.track import TrackPolicy
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import is_upright

logger = logging.getLogger("g1_playground")
RAMP_SECONDS = 3.0
BLEND_SECONDS = 5.0
IDLE_CONTROL = {"axes": {}}


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


@hydra.main(version_base=None, config_path="../configs", config_name="run_track")
def run(cfg: DictConfig) -> None:
    setup_logger()
    env = None
    try:
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        policy = TrackPolicy(cfg.policy, dof_cfg=dof)
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

        logger.warning("Ramping to the tracking standing pose over %.1f seconds", RAMP_SECONDS)
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

        logger.warning("Blending into closed-loop standing over %.1f seconds", BLEND_SECONDS)
        blend_steps = int(BLEND_SECONDS * policy.freq)
        for index in range(blend_steps):
            started = time.monotonic()
            frame = read_frame(env, controller)
            if frame is None:
                return
            state, _ = frame
            policy_target = policy.act(state, IDLE_CONTROL)
            alpha = (index + 1) / blend_steps
            env.step((1.0 - alpha) * standing + alpha * policy_target)
            remaining = policy.dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -policy.dt:
                logger.warning("Control frame dropped during policy blend")

        while True:
            started = time.monotonic()
            if not step(env, controller, policy):
                break
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
        if env is not None:
            env.shutdown()


if __name__ == "__main__":
    run()
