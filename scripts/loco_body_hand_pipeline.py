# ruff: noqa: I001
import os
import platform

if platform.machine() == "aarch64":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging
import select
import sys
import termios
import time
import tty
from types import SimpleNamespace

import g1_playground  # noqa: F401

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g1_playground.inspire.hand_env import InspireHandEnv
from g1_playground.policy import LeggedLabPolicy
from g1_playground.policy.body_hand import BodyHandPolicy
from g1_playground.policy.track import TrackPolicy
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import is_upright, quat_angular_velocity, quat_slerp, yaw_quat
from g1_playground.utils.recorder import record, recorder, save_recording

logger = logging.getLogger("g1_playground")
SWITCH_KEY = b"]"
WARMUP_STEPS = 3
HANDOVER_STABLE_FRAMES = 10
ZERO_CONTROL = {"axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0}}
IDLE_CONTROL = {"axes": {}}
PHASES = (
    "ramp",
    "blend",
    "locomotion",
    "to_largebox",
    "largebox",
    "largebox_settle",
    "to_standing",
    "stand_track",
)
LARGEBOX_PHASES = ("largebox", "largebox_settle")
STANDING_PHASES = ("to_standing", "stand_track")
TRACK_PHASES = ("to_largebox", *STANDING_PHASES)


class Stop(RuntimeError):
    pass


def open_key_reader():
    if not sys.stdin.isatty():
        logger.warning("stdin is not a terminal; the %r switch key is disabled", SWITCH_KEY.decode())
        return -1, None
    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    tty.setcbreak(descriptor)
    logger.warning("Terminal echo is off; press %r to hand over to the tracking policy", SWITCH_KEY.decode())
    return descriptor, saved


def poll_switch_key(descriptor: int):
    pressed = False
    while descriptor >= 0 and select.select([descriptor], [], [], 0.0)[0]:
        chunk = os.read(descriptor, 256)
        if not chunk:
            logger.warning("Standard input closed; the switch key is no longer available")
            return pressed, -1
        pressed = pressed or SWITCH_KEY in chunk
    return pressed, descriptor


def anchor_quat(state) -> np.ndarray:
    return np.asarray(state.base_quat, dtype=np.float32)[[3, 0, 1, 2]]


def eased(progress: float) -> float:
    return float(progress * progress * (3.0 - 2.0 * progress))


def read_frame(run):
    state = run.env.read()
    hand_state = run.hand_env.read()
    odometry = run.env.read_odometry()
    control, shutdown_requested = run.controller.read()
    if shutdown_requested:
        raise Stop("Emergency shutdown requested by the operator")
    if not is_upright(state.base_quat):
        raise Stop("Robot fallen")
    return state, hand_state, odometry, control


def require_largebox_inputs(hand_state, odometry) -> None:
    if odometry is None:
        raise Stop("Odometry is unavailable; the largebox reference cannot be anchored")
    if hand_state.stale:
        raise Stop(f"Inspire hand state is stale ({hand_state.age:.3f}s)")


def largebox_command(run, frame, state, hand_state):
    observation = run.largebox.get_observation(frame, anchor_quat(state), state, hand_state)
    return run.largebox.act(observation)


def rate_limited(previous, desired, limit):
    return previous + np.clip(desired - previous, -limit, limit)


def commit(run, phase, frame, state, hand_state, odometry, body_target, hand_target) -> None:
    run.env.step(body_target)
    run.hand_env.step(hand_target)
    record_frame(run, phase, frame, state, hand_state, odometry, body_target, hand_target)
    remaining = run.dt - (time.monotonic() - run.started)
    if remaining > 0:
        time.sleep(remaining)


def record_frame(run, phase, frame, state, hand_state, odometry, body_target, hand_target) -> None:
    log = run.log
    if log is None or log.count >= log.phase.shape[0]:
        return
    index = log.count
    log.phase[index] = PHASES.index(phase)
    log.motion_frame[index] = -1 if frame is None else frame
    log.hand_pos[index] = hand_state.joint_pos
    log.hand_target[index] = hand_target
    if phase in LARGEBOX_PHASES:
        log.largebox_action[index] = run.largebox.last_action
    elif phase in TRACK_PHASES:
        log.stand_action[index] = run.stand_track.last_action
    record(log, run.started - run.origin, state, body_target, odometry, run.env)


def warm_up_policies(run) -> None:
    state = run.env.read()
    largebox_observation = np.zeros(run.largebox.observation_dim, dtype=np.float32)

    loco_times = []
    largebox_times = []
    stand_times = []

    for _ in range(WARMUP_STEPS):
        started = time.perf_counter()
        run.loco.act(state, ZERO_CONTROL)
        loco_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        run.largebox.act(largebox_observation)
        largebox_times.append(time.perf_counter() - started)

        started = time.perf_counter()
        run.stand_track.act(state, IDLE_CONTROL)
        stand_times.append(time.perf_counter() - started)

    run.loco.reset()
    run.largebox.reset()
    run.stand_track.reset()

    logger.info(
        "Policy warm-up complete: loco %.1f -> %.1f ms, largebox %.1f -> %.1f ms, stand_track %.1f -> %.1f ms",
        loco_times[0] * 1000,
        loco_times[-1] * 1000,
        largebox_times[0] * 1000,
        largebox_times[-1] * 1000,
        stand_times[0] * 1000,
        stand_times[-1] * 1000,
    )


def startup_locomotion(run, ramp_steps, blend_steps):
    run.started = time.monotonic()
    state, hand_state, odometry, _ = read_frame(run)
    measured_body = np.asarray(state.dof_pos, dtype=np.float64)
    hand_command = np.asarray(hand_state.joint_pos, dtype=np.float64)
    standing = run.loco.standing_target

    run.env.activate_commands()
    run.origin = time.monotonic()
    logger.warning("Ramping to the locomotion standing pose over %.1f seconds", ramp_steps * run.dt)
    for index in range(ramp_steps):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        alpha = (index + 1) / ramp_steps
        body_command = (1.0 - alpha) * measured_body + alpha * standing
        commit(run, "ramp", None, state, hand_state, odometry, body_command, hand_command)

    run.loco.reset()
    logger.warning("Blending into closed-loop locomotion over %.1f seconds", blend_steps * run.dt)
    body_command = standing
    for index in range(blend_steps):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        alpha = (index + 1) / blend_steps
        body_command = (1.0 - alpha) * standing + alpha * run.loco.act(state, ZERO_CONTROL)
        commit(run, "blend", None, state, hand_state, odometry, body_command, hand_command)
    return body_command, hand_command


def run_locomotion_until_switch(run, body_command, hand_command):
    logger.warning("Locomotion is live; drive with the controller and press %r when parked", SWITCH_KEY.decode())
    descriptor = run.key_descriptor
    while True:
        run.started = time.monotonic()
        state, hand_state, odometry, control = read_frame(run)
        requested, descriptor = poll_switch_key(descriptor)
        run.key_descriptor = descriptor
        if requested:
            return body_command, hand_command, state, hand_state, odometry
        body_command = run.loco.act(state, control)
        commit(run, "locomotion", None, state, hand_state, odometry, body_command, hand_command)


def settle_locomotion_for_handover(run, steps, body_command, hand_command):
    """Keep locomotion closed-loop at zero command until the base is stably parked."""
    logger.warning("Waiting up to %.1f seconds for a stable zero-command handover", steps * run.dt)
    stable_frames = 0
    for _ in range(steps):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        require_largebox_inputs(hand_state, odometry)
        stable = (
            is_upright(state.base_quat, max_tilt=run.handover_max_tilt)
            and float(odometry.body_height) >= run.handover_min_height
            and float(np.linalg.norm(odometry.velocity)) <= run.handover_max_linear_speed
            and float(np.linalg.norm(state.base_ang_vel)) <= run.handover_max_angular_speed
        )
        stable_frames = stable_frames + 1 if stable else 0
        if stable_frames >= HANDOVER_STABLE_FRAMES:
            logger.warning(
                "Locomotion handover is stable: height %.3f m, linear speed %.3f m/s, angular speed %.3f rad/s",
                float(odometry.body_height),
                float(np.linalg.norm(odometry.velocity)),
                float(np.linalg.norm(state.base_ang_vel)),
            )
            return body_command, hand_command, state, hand_state, odometry

        body_command = run.loco.act(state, ZERO_CONTROL)
        commit(run, "locomotion", None, state, hand_state, odometry, body_command, hand_command)
    raise Stop("Locomotion did not reach a stable handover state; keep the robot parked before switching")


def capture_largebox_origin(run, state, hand_state, odometry):
    require_largebox_inputs(hand_state, odometry)
    logger.warning(
        "Capturing the track origin at raw odometry xy=%s, body height %.3f m",
        np.asarray(odometry.raw_position)[:2],
        odometry.body_height,
    )
    run.env.set_born_place(state.base_quat, odometry.raw_position)
    run.largebox.motion.align()
    run.largebox.reset()

    state = run.env.read()
    odometry = run.env.read_odometry()
    require_largebox_inputs(hand_state, odometry)
    logger.warning(
        "Rebased: local xy=%s, body height %.3f m, reference frame 0 at xy=%s",
        np.asarray(odometry.position)[:2],
        float(odometry.position[2]),
        run.largebox.motion.anchor_pos[0][:2],
    )
    return state, odometry


def blend_to_largebox(run, steps, body_command, hand_command):
    """Use the standing tracker to move to frame 0 without running a frozen HOI policy."""
    if steps <= 0:
        raise ValueError("The locomotion-to-HOI reference transition must contain at least one step")
    run.env.set_gains(run.largebox_stiffness, run.largebox_damping)
    logger.warning(
        "Switched to the tracking gains; walking the reference to HOI frame 0 over %.1f seconds",
        steps * run.dt,
    )

    state, hand_state, odometry, _ = read_frame(run)
    require_largebox_inputs(hand_state, odometry)
    start = SimpleNamespace(
        joint_pos=np.asarray(state.dof_pos, dtype=np.float32).copy(),
        root_quat=anchor_quat(state),
        root_height=float(odometry.position[2]),
        hand_pos=np.asarray(hand_command, dtype=np.float64).copy(),
    )
    target_body, target_hand = run.largebox.reference_targets()
    target_body = np.asarray(target_body, dtype=np.float32)
    target_hand = np.asarray(target_hand, dtype=np.float64)
    target_quat = np.asarray(run.largebox.motion.anchor_quat[0], dtype=np.float32)
    target_height = float(run.largebox.motion.anchor_pos[0, 2])

    run.stand_track.reset()
    run.stand_track.accept_applied_target(body_command)
    previous = apply_reference(
        run,
        start.root_height,
        start.root_quat,
        start.joint_pos,
        np.zeros(29, dtype=np.float32),
        None,
    )
    duration = steps * run.dt
    for index in range(steps):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        require_largebox_inputs(hand_state, odometry)

        progress = (index + 1) / steps
        alpha = eased(progress)
        alpha_rate = 6.0 * progress * (1.0 - progress) / duration
        previous = apply_reference(
            run,
            (1.0 - alpha) * start.root_height + alpha * target_height,
            quat_slerp(start.root_quat, target_quat, alpha),
            (1.0 - alpha) * start.joint_pos + alpha * target_body,
            alpha_rate * (target_body - start.joint_pos),
            previous,
        )

        policy_target = run.stand_track.act(state, IDLE_CONTROL)
        body_command = rate_limited(body_command, policy_target, run.body_rate_limit)
        run.stand_track.accept_applied_target(body_command)
        hand_command = (1.0 - alpha) * start.hand_pos + alpha * target_hand
        commit(run, "to_largebox", None, state, hand_state, odometry, body_command, hand_command)

    apply_reference(
        run,
        target_height,
        target_quat,
        target_body,
        np.zeros(29, dtype=np.float32),
        None,
    )
    run.largebox.reset()
    return body_command, hand_command


def run_largebox_motion(run):
    frames = run.largebox.motion.num_frames
    source_frames = getattr(run.largebox.motion, "source_num_frames", frames)
    terminal_frames = getattr(run.largebox.motion, "terminal_hold_frames", 0)
    logger.warning(
        "Running %d source frames plus %d terminal hold frames once (%.2f s)",
        source_frames,
        terminal_frames,
        frames * run.dt,
    )
    for frame in range(frames):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        require_largebox_inputs(hand_state, odometry)
        body_command, hand_command = largebox_command(run, frame, state, hand_state)
        commit(run, "largebox", frame, state, hand_state, odometry, body_command, hand_command)
    return body_command, hand_command


def settle_largebox_for_return(run, steps, body_command, hand_command):
    """Keep the static terminal reference closed-loop until a return handoff is genuinely stable."""
    if steps <= 0:
        raise ValueError("The HOI return settle window must contain at least one step")
    frame = run.largebox.motion.num_frames - 1
    stable_frames = 0
    last_metrics = None
    logger.warning("Waiting up to %.1f seconds for a stable HOI-to-standing handover", steps * run.dt)
    for _ in range(steps):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        require_largebox_inputs(hand_state, odometry)
        last_metrics = (
            float(odometry.body_height),
            float(np.linalg.norm(odometry.velocity)),
            float(np.linalg.norm(state.base_ang_vel)),
            float(np.max(np.abs(state.dof_vel))),
        )
        stable = (
            is_upright(state.base_quat, max_tilt=run.handover_max_tilt)
            and last_metrics[0] >= run.handover_min_height
            and last_metrics[1] <= run.handover_max_linear_speed
            and last_metrics[2] <= run.handover_max_angular_speed
            and last_metrics[3] <= run.return_max_joint_speed
        )
        stable_frames = stable_frames + 1 if stable else 0
        if stable_frames >= HANDOVER_STABLE_FRAMES:
            logger.warning(
                "HOI return handover is stable: height %.3f m, linear %.3f m/s, angular %.3f rad/s, "
                "max joint speed %.3f rad/s",
                *last_metrics,
            )
            return body_command, hand_command, state, hand_state, odometry

        desired_body, desired_hand = largebox_command(run, frame, state, hand_state)
        body_command = rate_limited(body_command, desired_body, run.body_rate_limit)
        hand_command = desired_hand
        commit(run, "largebox_settle", frame, state, hand_state, odometry, body_command, hand_command)

    height, linear, angular, joint = last_metrics
    raise Stop(
        "HOI terminal reference did not settle safely "
        f"(height {height:.3f} m, linear {linear:.3f} m/s, angular {angular:.3f} rad/s, "
        f"max joint speed {joint:.3f} rad/s)"
    )


def reference_anchor_pose(run, root_height, root_quat, joint_pos):
    return run.stand_track.observation.anchor_pose(
        np.array([0.0, 0.0, float(root_height)], dtype=np.float32),
        np.asarray(root_quat, dtype=np.float32).reshape(4),
        np.asarray(joint_pos, dtype=np.float32).reshape(-1),
    )


def apply_reference(run, root_height, root_quat, joint_pos, joint_vel, previous):
    anchor_pos, anchor_quat = reference_anchor_pose(run, root_height, root_quat, joint_pos)
    if previous is None:
        anchor_lin_vel_w = np.zeros(3, dtype=np.float32)
        anchor_ang_vel_w = np.zeros(3, dtype=np.float32)
    else:
        anchor_lin_vel_w = (anchor_pos - previous[0]) / run.dt
        anchor_ang_vel_w = quat_angular_velocity(previous[1], anchor_quat, run.dt)
    run.stand_track.set_reference(
        root_height=root_height,
        root_quat=root_quat,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        anchor_lin_vel_w=anchor_lin_vel_w,
        anchor_ang_vel_w=anchor_ang_vel_w,
    )
    return anchor_pos, anchor_quat


def capture_standing_reference(run, state, odometry, applied_body_command):
    measured_body = np.asarray(state.dof_pos, dtype=np.float64).copy()
    applied_body_command = np.asarray(applied_body_command, dtype=np.float64).copy()
    run.stand_track.reset()
    run.stand_track.accept_applied_target(applied_body_command)
    start = SimpleNamespace(
        joint_pos=measured_body.astype(np.float32),
        root_quat=anchor_quat(state),
        root_height=float(odometry.position[2]),
    )
    start.anchor = apply_reference(
        run, start.root_height, start.root_quat, start.joint_pos, np.zeros(29, dtype=np.float32), None
    )
    logger.warning(
        "Standing reference captured live: root height %.3f m, %.2f rad from the default pose",
        start.root_height,
        float(np.abs(start.joint_pos - run.stand_track.standing_target).max()),
    )
    return start


def track_to_standing(run, steps, body_command, hand_command, start):
    default_pos = np.asarray(run.stand_track.standing_target, dtype=np.float32)
    target_height = float(run.stand_track.reference_root_height)
    target_quat = yaw_quat(start.root_quat)
    duration = steps * run.dt
    logger.warning(
        "Walking the standing reference to the default pose over %.1f seconds (%.2f rad, %+.3f m)",
        duration,
        float(np.abs(default_pos - start.joint_pos).max()),
        target_height - start.root_height,
    )
    previous = start.anchor
    for index in range(steps):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)

        progress = (index + 1) / steps
        alpha = eased(progress)
        alpha_rate = 6.0 * progress * (1.0 - progress) / duration
        previous = apply_reference(
            run,
            (1.0 - alpha) * start.root_height + alpha * target_height,
            quat_slerp(start.root_quat, target_quat, alpha),
            (1.0 - alpha) * start.joint_pos + alpha * default_pos,
            alpha_rate * (default_pos - start.joint_pos),
            previous,
        )

        policy_target = run.stand_track.act(state, IDLE_CONTROL)
        body_command = rate_limited(body_command, policy_target, run.body_rate_limit)
        run.stand_track.accept_applied_target(body_command)
        commit(run, "to_standing", None, state, hand_state, odometry, body_command, hand_command)

    apply_reference(run, target_height, target_quat, default_pos, np.zeros(29, dtype=np.float32), None)
    return body_command, hand_command


def hold_stand_track(run, body_command, hand_command):
    logger.warning("Standing tracker holds the default reference; press the shutdown button to finish")
    while True:
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        policy_target = run.stand_track.act(state, IDLE_CONTROL)
        body_command = rate_limited(body_command, policy_target, run.body_rate_limit)
        run.stand_track.accept_applied_target(body_command)
        commit(run, "stand_track", None, state, hand_state, odometry, body_command, hand_command)


def build_log(capacity, largebox, stand_track, hand_dofs):
    log = recorder(capacity)
    log.phase = np.full(capacity, -1, dtype=np.int8)
    log.motion_frame = np.full(capacity, -1, dtype=np.int32)
    log.hand_pos = np.full((capacity, hand_dofs), np.nan, dtype=np.float32)
    log.hand_target = np.full((capacity, hand_dofs), np.nan, dtype=np.float32)
    log.largebox_action = np.full((capacity, largebox.action_dim), np.nan, dtype=np.float32)
    log.stand_action = np.full((capacity, stand_track.standing_target.shape[0]), np.nan, dtype=np.float32)
    log.phase_names = np.asarray(PHASES, dtype="<U16")
    return log


@hydra.main(version_base=None, config_path="../configs", config_name="run_loco_largebox_track")
def run(cfg: DictConfig) -> None:
    setup_logger()
    torch.set_num_threads(1)
    env = None
    hand_env = None
    log = None
    saved_terminal = None
    try:
        loco_dof = compose_dof_config(cfg.robot.dof, cfg.loco.dof)
        stand_dof = compose_dof_config(cfg.robot.dof, cfg.stand_track.dof)
        loco = LeggedLabPolicy(cfg.loco, device=cfg.device, dof_cfg=loco_dof)
        largebox = BodyHandPolicy(
            cfg.largebox,
            cfg.motion,
            device=cfg.device,
            runtime_body_joint_names=cfg.robot.dof.joint_names,
            runtime_hand_joint_names=cfg.inspire.dof.joint_names,
            hand_mimic=cfg.inspire.mimic,
        )
        stand_track = TrackPolicy(cfg.stand_track, device=cfg.device, dof_cfg=stand_dof)
        if not loco.freq == largebox.freq == stand_track.freq:
            raise RuntimeError(
                f"All policies must run at one rate: loco {loco.freq} Hz, largebox {largebox.freq} Hz, "
                f"stand_track {stand_track.freq} Hz"
            )

        env = instantiate(cfg.env, dof_cfg=loco_dof, control_dt=loco.dt)
        hand_env = InspireHandEnv(dof_cfg=cfg.inspire.dof, domain_id=cfg.env.domain_id, net_if=cfg.env.net_if)
        controller = instantiate(cfg.controller, env=env)

        body_control = cfg.largebox.action.body.control
        run = SimpleNamespace(
            env=env,
            hand_env=hand_env,
            controller=controller,
            loco=loco,
            largebox=largebox,
            stand_track=stand_track,
            dt=loco.dt,
            log=None,
            started=0.0,
            origin=0.0,
            key_descriptor=-1,
            body_rate_limit=float(cfg.handover.body_rate_limit),
            handover_min_height=float(cfg.handover.min_body_height),
            handover_max_tilt=float(cfg.handover.max_body_tilt),
            handover_max_linear_speed=float(cfg.handover.max_linear_speed),
            handover_max_angular_speed=float(cfg.handover.max_angular_speed),
            return_max_joint_speed=float(cfg.handover.return_max_joint_speed),
            largebox_stiffness=largebox.body_to_runtime.fit(body_control.stiffness),
            largebox_damping=largebox.body_to_runtime.fit(body_control.damping),
        )
        if cfg.recording.enabled:
            log = build_log(int(cfg.recording.seconds * loco.freq), largebox, stand_track, hand_env.num_dofs)
            run.log = log

        env.self_check()
        hand_env.self_check()

        warm_up_policies(run)

        run.key_descriptor, saved_terminal = open_key_reader()

        ramp_steps = int(cfg.startup.ramp_seconds * loco.freq)
        blend_steps = int(cfg.startup.blend_seconds * loco.freq)
        settle_steps = int(cfg.handover.settle_seconds * loco.freq)
        to_largebox_steps = int(cfg.handover.to_largebox_seconds * loco.freq)
        return_settle_steps = int(cfg.handover.return_settle_seconds * loco.freq)
        to_standing_steps = int(cfg.handover.to_standing_seconds * loco.freq)

        body_command, hand_command = startup_locomotion(run, ramp_steps, blend_steps)
        body_command, hand_command, state, hand_state, odometry = run_locomotion_until_switch(
            run, body_command, hand_command
        )
        body_command, hand_command, state, hand_state, odometry = settle_locomotion_for_handover(
            run, settle_steps, body_command, hand_command
        )
        state, odometry = capture_largebox_origin(run, state, hand_state, odometry)
        body_command, hand_command = blend_to_largebox(run, to_largebox_steps, body_command, hand_command)
        body_command, hand_command = run_largebox_motion(run)
        body_command, hand_command, state, hand_state, odometry = settle_largebox_for_return(
            run, return_settle_steps, body_command, hand_command
        )
        logger.warning(
            "HOI terminal tracking error is %.2f rad; preserving the applied target for a continuous handover",
            float(np.abs(np.asarray(body_command, dtype=np.float64) - np.asarray(state.dof_pos)).max()),
        )
        start = capture_standing_reference(run, state, odometry, body_command)
        body_command, hand_command = track_to_standing(run, to_standing_steps, body_command, hand_command, start)
        hold_stand_track(run, body_command, hand_command)
    except Stop as stopped:
        logger.critical("Stopping: %s", stopped)
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        if saved_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved_terminal)
        if hand_env is not None:
            hand_env.shutdown()
        if env is not None:
            env.shutdown()
        if log is not None:
            save_recording(log, cfg.recording.directory, OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    run()
