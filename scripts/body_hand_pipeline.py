# ruff: noqa: I001
import os
import platform

if platform.machine() == "aarch64":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import time

import g1_playground  # noqa: F401

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g1_playground.inspire.hand_env import InspireHandEnv
from g1_playground.policy.body_hand import BodyHandPolicy
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.recorder import record, recorder, save_recording

logger = logging.getLogger("g1_playground")


def read_frame(env, hand_env):
    state = env.read()
    hand_state = hand_env.read()
    odometry = env.read_odometry()
    if odometry is None:
        raise RuntimeError("Odometry is unavailable")
    if hand_state.stale:
        raise RuntimeError(f"Inspire hand state is stale ({hand_state.age:.3f}s)")
    return state, hand_state, odometry


def anchor_quat(state) -> np.ndarray:
    return np.asarray(state.base_quat, dtype=np.float32)[[3, 0, 1, 2]]


def send_command(env, hand_env, body_target, hand_target) -> None:
    env.step(body_target)
    hand_env.step(hand_target)


def pace(started: float, dt: float) -> None:
    remaining = dt - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)


def policy_command(policy, frame, state, hand_state, odometry):
    observation = policy.get_observation(frame, odometry.position, anchor_quat(state), state, hand_state)
    return policy.act(observation)


def ramp_to_reference(env, hand_env, policy, steps, measured_body, measured_hand):
    reference_body, reference_hand = policy.reference_targets()
    logger.warning("Ramping to the first reference pose over %.1f seconds", steps * policy.dt)
    for index in range(steps):
        started = time.monotonic()
        read_frame(env, hand_env)
        alpha = (index + 1) / steps
        send_command(
            env,
            hand_env,
            (1.0 - alpha) * measured_body + alpha * reference_body,
            (1.0 - alpha) * measured_hand + alpha * reference_hand,
        )
        pace(started, policy.dt)
    return reference_body, reference_hand


def capture_origin(env, hand_env, policy) -> None:
    state, _, odometry = read_frame(env, hand_env)
    env.set_born_place(state.base_quat, odometry.position)
    policy.motion.align()
    policy.reset()
    logger.warning("Reference aligned to the current robot pose")


def blend_into_policy(env, hand_env, policy, steps, reference_body, reference_hand) -> None:
    logger.warning("Blending into the closed-loop policy over %.1f seconds", steps * policy.dt)
    for index in range(steps):
        started = time.monotonic()
        state, hand_state, odometry = read_frame(env, hand_env)
        policy_body, policy_hand = policy_command(policy, 0, state, hand_state, odometry)
        alpha = (index + 1) / steps
        send_command(
            env,
            hand_env,
            (1.0 - alpha) * reference_body + alpha * policy_body,
            (1.0 - alpha) * reference_hand + alpha * policy_hand,
        )
        pace(started, policy.dt)


def record_policy_frame(log, elapsed, state, hand_state, odometry, body_target, hand_target, policy, env) -> None:
    if log is None:
        return
    index = log.count
    log.raw_action[index] = policy.last_action
    log.hand_pos[index] = hand_state.joint_pos
    log.hand_vel[index] = hand_state.joint_vel
    log.hand_target[index] = hand_target
    record(log, elapsed, state, body_target, odometry, env)


def run_motion(env, hand_env, policy, log) -> None:
    logger.warning("Running all %d reference frames once", policy.motion.num_frames)
    origin = time.monotonic()
    for frame in range(policy.motion.num_frames):
        started = time.monotonic()
        state, hand_state, odometry = read_frame(env, hand_env)
        body_target, hand_target = policy_command(policy, frame, state, hand_state, odometry)
        send_command(env, hand_env, body_target, hand_target)
        record_policy_frame(
            log,
            started - origin,
            state,
            hand_state,
            odometry,
            body_target,
            hand_target,
            policy,
            env,
        )
        pace(started, policy.dt)


@hydra.main(version_base=None, config_path="../configs", config_name="run_body_hand")
def run(cfg: DictConfig) -> None:
    setup_logger()
    env = None
    hand_env = None
    log = None
    try:
        inspire = OmegaConf.load(resolve_repo_path("configs/robot/inspire.yaml"))
        policy = BodyHandPolicy(
            cfg.policy,
            cfg.motion,
            device=cfg.device,
            runtime_body_joint_names=cfg.robot.dof.joint_names,
            runtime_hand_joint_names=inspire.dof.joint_names,
            hand_mimic=inspire.mimic,
        )

        body_control = cfg.policy.action.body.control
        dof = OmegaConf.create(
            {
                "joint_names": list(cfg.robot.dof.joint_names),
                "stiffness": policy.body_to_runtime.fit(body_control.stiffness).tolist(),
                "damping": policy.body_to_runtime.fit(body_control.damping).tolist(),
            }
        )
        env = instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)
        hand_env = InspireHandEnv(dof_cfg=inspire.dof, domain_id=cfg.env.domain_id, net_if=cfg.env.net_if)

        if cfg.recording.enabled:
            capacity = policy.motion.num_frames
            log = recorder(capacity)
            log.raw_action = np.empty((capacity, policy.action_dim), dtype=np.float32)
            log.hand_pos = np.empty((capacity, hand_env.num_dofs), dtype=np.float32)
            log.hand_vel = np.empty((capacity, hand_env.num_dofs), dtype=np.float32)
            log.hand_target = np.empty((capacity, hand_env.num_dofs), dtype=np.float32)

        env.self_check()
        hand_env.self_check()
        state, hand_state, _ = read_frame(env, hand_env)
        env.activate_commands()

        reference_body, reference_hand = ramp_to_reference(
            env,
            hand_env,
            policy,
            int(cfg.startup.ramp_seconds * policy.freq),
            state.dof_pos,
            hand_state.joint_pos,
        )
        capture_origin(env, hand_env, policy)
        blend_into_policy(
            env,
            hand_env,
            policy,
            int(cfg.startup.blend_seconds * policy.freq),
            reference_body,
            reference_hand,
        )
        run_motion(env, hand_env, policy, log)
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        if log is not None:
            save_recording(log, cfg.recording.directory, OmegaConf.to_yaml(cfg, resolve=True))
        if hand_env is not None:
            try:
                hand_env.open()
            except Exception as error:
                logger.error("Failed to open the hand: %r", error)
            hand_env.shutdown()
        if env is not None:
            env.shutdown()


if __name__ == "__main__":
    run()
