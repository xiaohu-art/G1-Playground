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
from enum import Enum, auto
from types import SimpleNamespace

import g1_playground  # noqa: F401

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g1_playground.inspire.hand_env import InspireHandEnv
from g1_playground.inspire.service import InspireService
from g1_playground.policy import LeggedLabPolicy
from g1_playground.policy.body_hand import BodyHandPolicy
from g1_playground.policy.body_hand.depth import DepthPreviewServer
from g1_playground.policy.body_hand.motion import select_motion
from g1_playground.policy.track import TrackPolicy
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import is_upright, quat_angular_velocity, quat_slerp, yaw_quat
from g1_playground.utils.recorder import record, recorder, save_recording

logger = logging.getLogger("g1_playground")
LOCO_TO_TRACK_KEY = b"["
TRACK_TO_HOI_KEY = b"]"
WARMUP_STEPS = 3
ZERO_CONTROL = {"axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0}}
IDLE_CONTROL = {"axes": {}}
PHASES = ("ramp", "locomotion", "track", "track_to_hoi", "hoi", "track_to_default")


class Mode(Enum):
    LOCOMOTION = auto()
    TRACK = auto()
    HOI = auto()


class Stop(RuntimeError):
    pass


def open_key_reader():
    if not sys.stdin.isatty():
        logger.warning("stdin is not a terminal; the '[' and ']' switch keys are disabled")
        return -1, None
    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    tty.setcbreak(descriptor)
    logger.warning("Terminal echo is off; press '[' to prepare HOI frame 0 with Track, then ']' to run HOI")
    return descriptor, saved


def poll_keys(descriptor: int):
    pressed = set()
    while descriptor >= 0 and select.select([descriptor], [], [], 0.0)[0]:
        chunk = os.read(descriptor, 256)
        if not chunk:
            logger.warning("Standard input closed; the switch keys are no longer available")
            return pressed, -1
        for key in (LOCO_TO_TRACK_KEY, TRACK_TO_HOI_KEY):
            if key in chunk:
                pressed.add(key)
    return pressed, descriptor


def smoothstep(progress: float) -> float:
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


def record_frame(
    run,
    phase,
    frame,
    state,
    hand_state,
    odometry,
    body_target,
    hand_target,
    depth_diagnostics=None,
) -> None:
    log = run.log
    if log is None or log.count >= log.phase.shape[0]:
        return
    index = log.count
    log.phase[index] = PHASES.index(phase)
    log.motion_frame[index] = -1 if frame is None else frame
    log.hand_pos[index] = hand_state.joint_pos
    log.hand_target[index] = hand_target
    if depth_diagnostics is not None:
        (
            log.depth_sequence[index],
            log.depth_age[index],
            log.depth_valid_ratio[index],
            log.depth_min[index],
            log.depth_max[index],
        ) = depth_diagnostics
    record(log, run.started - run.origin, state, body_target, odometry, run.env)


def step_envs(run, phase, frame, state, hand_state, odometry, body_target, hand_target, depth_diagnostics=None) -> None:
    run.env.step(body_target)
    run.hand_env.step(hand_target)
    record_frame(
        run,
        phase,
        frame,
        state,
        hand_state,
        odometry,
        body_target,
        hand_target,
        depth_diagnostics,
    )
    remaining = run.dt - (time.monotonic() - run.started)
    if remaining > 0:
        time.sleep(remaining)


def warm_up_policies(run) -> None:
    state = run.env.read()
    hoi_observation = np.zeros(run.hoi.observation_dim, dtype=np.float32)
    timings = {"loco": [], "hoi": [], "track": []}

    for _ in range(WARMUP_STEPS):
        started = time.perf_counter()
        run.loco.act(state, ZERO_CONTROL)
        timings["loco"].append(time.perf_counter() - started)

        started = time.perf_counter()
        run.hoi.act(hoi_observation)
        timings["hoi"].append(time.perf_counter() - started)

        started = time.perf_counter()
        run.track.act(state, IDLE_CONTROL)
        timings["track"].append(time.perf_counter() - started)

    run.loco.reset()
    run.hoi.reset()
    run.track.reset()

    logger.info(
        "Policy warm-up complete: loco %.1f -> %.1f ms, HOI %.1f -> %.1f ms, track %.1f -> %.1f ms",
        timings["loco"][0] * 1000,
        timings["loco"][-1] * 1000,
        timings["hoi"][0] * 1000,
        timings["hoi"][-1] * 1000,
        timings["track"][0] * 1000,
        timings["track"][-1] * 1000,
    )


def startup_locomotion(run, ramp_steps):
    run.started = time.monotonic()
    state, hand_state, _, _ = read_frame(run)
    measured_body = np.asarray(state.dof_pos, dtype=np.float64)
    hand_command = np.asarray(hand_state.joint_pos, dtype=np.float64)
    standing_target = run.loco.standing_target

    run.env.activate_commands()
    run.origin = time.monotonic()
    logger.warning("Ramping to the locomotion standing pose over %.1f seconds", ramp_steps * run.dt)
    for index in range(ramp_steps):
        run.started = time.monotonic()
        state, hand_state, odometry, _ = read_frame(run)
        alpha = (index + 1) / ramp_steps
        body_command = (1.0 - alpha) * measured_body + alpha * standing_target
        step_envs(run, "ramp", None, state, hand_state, odometry, body_command, hand_command)

    run.loco.reset()
    return standing_target, hand_command


def apply_track_reference(run, reference, joint_vel, previous_anchor=None):
    anchor_pos, anchor_orientation = run.track.observation.anchor_pose(
        np.array([0.0, 0.0, reference.root_height], dtype=np.float32),
        reference.root_quat,
        reference.joint_pos,
    )
    if previous_anchor is None:
        anchor_lin_vel_w = np.zeros(3, dtype=np.float32)
        anchor_ang_vel_w = np.zeros(3, dtype=np.float32)
    else:
        anchor_lin_vel_w = (anchor_pos - previous_anchor[0]) / run.dt
        anchor_ang_vel_w = quat_angular_velocity(previous_anchor[1], anchor_orientation, run.dt)
    run.track.set_reference(
        root_height=reference.root_height,
        root_quat=reference.root_quat,
        joint_pos=reference.joint_pos,
        joint_vel=joint_vel,
        anchor_lin_vel_w=anchor_lin_vel_w,
        anchor_ang_vel_w=anchor_ang_vel_w,
    )
    reference.anchor = (anchor_pos, anchor_orientation)
    return reference


def enter_track(run, machine, state, odometry):
    if odometry is None:
        raise Stop("Odometry is unavailable; cannot capture the Track origin")
    logger.warning(
        "Capturing the track origin at raw odometry xy=%s, body height %.3f m",
        np.asarray(odometry.raw_position)[:2],
        odometry.body_height,
    )
    run.env.set_born_place(state.base_quat, odometry.raw_position)
    run.hoi.motion.align()
    run.env.set_gains(run.hoi_stiffness, run.hoi_damping)

    state = run.env.read()
    odometry = run.env.read_odometry()
    if odometry is None:
        raise Stop("Odometry is unavailable after rebase")

    run.track.reset()
    machine.track_reference = apply_track_reference(
        run,
        SimpleNamespace(
            root_height=float(odometry.position[2]),
            root_quat=np.asarray(state.base_quat, dtype=np.float32)[[3, 0, 1, 2]],
            joint_pos=np.asarray(state.dof_pos, dtype=np.float32).copy(),
            hand_pos=machine.hand_command.copy(),
        ),
        np.zeros(29, dtype=np.float32),
    )
    target_body, target_hand = run.hoi.reference_targets()
    machine.mode = Mode.TRACK
    machine.hoi_ready = False
    machine.track_goal = "hoi"
    machine.transition = SimpleNamespace(
        start=machine.track_reference,
        target=SimpleNamespace(
            root_height=float(run.hoi.motion.anchor_pos[0, 2]),
            root_quat=np.asarray(run.hoi.motion.anchor_quat[0], dtype=np.float32),
            joint_pos=np.asarray(target_body, dtype=np.float32),
            hand_pos=np.asarray(target_hand, dtype=np.float64),
        ),
        steps=machine.to_hoi_steps,
        step=0,
    )
    logger.warning(
        "Track captured the live reference at local xy=%s and body height %.3f m; "
        "moving to HOI frame 0 over %.1f seconds",
        np.asarray(odometry.position)[:2],
        float(odometry.position[2]),
        machine.to_hoi_steps * run.dt,
    )
    return state, odometry


def step_track_transition(run, machine):
    transition = machine.transition
    transition.step += 1
    progress = transition.step / transition.steps
    alpha = smoothstep(progress)
    alpha_rate = 6.0 * progress * (1.0 - progress) / (transition.steps * run.dt)
    start, target = transition.start, transition.target

    reference = SimpleNamespace(
        root_height=(1.0 - alpha) * start.root_height + alpha * target.root_height,
        root_quat=quat_slerp(start.root_quat, target.root_quat, alpha),
        joint_pos=(1.0 - alpha) * start.joint_pos + alpha * target.joint_pos,
        hand_pos=(1.0 - alpha) * start.hand_pos + alpha * target.hand_pos,
    )
    machine.track_reference = apply_track_reference(
        run,
        reference,
        alpha_rate * (target.joint_pos - start.joint_pos),
        machine.track_reference.anchor,
    )
    machine.hand_command = reference.hand_pos
    return transition.step == transition.steps


def run_pipeline(run, body_command, hand_command, to_hoi_steps, to_default_steps):
    machine = SimpleNamespace(
        mode=Mode.LOCOMOTION,
        track_goal=None,
        transition=None,
        track_reference=None,
        hoi_ready=False,
        motion_frame=0,
        body_command=np.asarray(body_command, dtype=np.float64),
        hand_command=np.asarray(hand_command, dtype=np.float64),
        to_hoi_steps=to_hoi_steps,
        to_default_steps=to_default_steps,
    )
    run.machine = machine
    logger.warning("Locomotion is live; press '[' when parked to hand control to Track")

    while True:
        run.started = time.monotonic()
        state, hand_state, odometry, control = read_frame(run)
        keys, run.key_descriptor = poll_keys(run.key_descriptor)
        frame = None
        depth_diagnostics = None
        depth_frame = None
        if run.depth_preview is not None or machine.mode is Mode.HOI:
            depth_frame = run.camera.read()
            if run.depth_preview is not None:
                run.depth_preview.publish(depth_frame)

        match machine.mode:
            case Mode.LOCOMOTION:
                if LOCO_TO_TRACK_KEY in keys:
                    state, odometry = enter_track(run, machine, state, odometry)
                    machine.body_command = run.track.act(state, IDLE_CONTROL)
                    phase = "track"
                else:
                    machine.body_command = run.loco.act(state, control)
                    phase = "locomotion"

            case Mode.TRACK:
                if machine.track_goal == "default" and machine.transition is None:
                    if odometry is None:
                        raise Stop("Odometry is unavailable; the live Track reference cannot be captured")
                    run.track.reset()
                    machine.track_reference = apply_track_reference(
                        run,
                        SimpleNamespace(
                            root_height=float(odometry.position[2]),
                            root_quat=np.asarray(state.base_quat, dtype=np.float32)[[3, 0, 1, 2]],
                            joint_pos=np.asarray(state.dof_pos, dtype=np.float32).copy(),
                            hand_pos=machine.hand_command.copy(),
                        ),
                        np.zeros(29, dtype=np.float32),
                    )
                    target = SimpleNamespace(
                        root_height=float(run.track.reference_root_height),
                        root_quat=yaw_quat(machine.track_reference.root_quat),
                        joint_pos=np.asarray(run.track.standing_target, dtype=np.float32),
                        hand_pos=machine.track_reference.hand_pos.copy(),
                    )
                    machine.transition = SimpleNamespace(
                        start=machine.track_reference,
                        target=target,
                        steps=machine.to_default_steps,
                        step=0,
                    )
                    logger.warning(
                        "HOI finished; Track is moving the live reference to the default pose over %.1f seconds "
                        "(%.2f rad, %+.3f m)",
                        machine.to_default_steps * run.dt,
                        float(np.abs(target.joint_pos - machine.track_reference.joint_pos).max()),
                        target.root_height - machine.track_reference.root_height,
                    )
                elif TRACK_TO_HOI_KEY in keys and machine.hoi_ready:
                    if hand_state.stale:
                        raise Stop(f"Inspire hand state is stale ({hand_state.age:.3f}s)")
                    run.hoi.reset()
                    machine.hoi_ready = False
                    machine.motion_frame = 0
                    machine.mode = Mode.HOI
                    logger.warning("Running all %d HOI reference frames once", run.hoi.motion.num_frames)

                if machine.track_goal == "hoi" and hand_state.stale:
                    raise Stop(f"Inspire hand state is stale ({hand_state.age:.3f}s)")

                goal = machine.track_goal
                finished = machine.transition is not None and step_track_transition(run, machine)
                machine.body_command = run.track.act(state, IDLE_CONTROL)
                phase = "track" if goal is None else f"track_to_{goal}"
                if finished:
                    target = machine.transition.target
                    machine.track_reference = apply_track_reference(
                        run,
                        target,
                        np.zeros_like(target.joint_pos),
                    )
                    machine.hand_command = target.hand_pos
                    machine.track_goal = None
                    machine.transition = None
                    if goal == "hoi":
                        machine.hoi_ready = True
                        logger.warning("Track holds HOI frame 0; press ']' to run HOI")
                    else:
                        logger.warning("Track holds the default reference; press the shutdown button to finish")

            case Mode.HOI:
                frame = machine.motion_frame
                if odometry is None:
                    raise Stop("Odometry is unavailable during HOI")
                if hand_state.stale:
                    raise Stop(f"Inspire hand state is stale ({hand_state.age:.3f}s)")
                depth_age = time.monotonic() - depth_frame.timestamp
                if not np.isfinite(depth_age) or depth_age > run.depth_stale_seconds:
                    raise Stop(f"Depth frame is stale ({depth_age:.3f}s)")
                depth_m = depth_frame.depth_m
                valid = (
                    np.isfinite(depth_m)
                    & (depth_m > run.hoi.observation.min_distance)
                    & (depth_m < run.hoi.observation.max_distance)
                )
                valid_depth = depth_m[valid]
                depth_diagnostics = (
                    depth_frame.sequence,
                    depth_age,
                    float(valid.mean()),
                    float(valid_depth.min()) if valid_depth.size else np.nan,
                    float(valid_depth.max()) if valid_depth.size else np.nan,
                )
                base_quat_wxyz = np.asarray(state.base_quat, dtype=np.float32)[[3, 0, 1, 2]]
                observation = run.hoi.get_observation(
                    frame,
                    odometry.position,
                    base_quat_wxyz,
                    state,
                    hand_state,
                    depth_m=depth_m,
                )
                body_target, hand_target = run.hoi.act(observation)
                if frame == 0:
                    logger.warning(
                        "Track-to-HOI first command step: %.3f rad max",
                        float(np.max(np.abs(body_target - machine.body_command))),
                    )
                machine.body_command = body_target
                machine.hand_command = hand_target
                machine.motion_frame += 1
                phase = "hoi"
                if machine.motion_frame == run.hoi.motion.num_frames:
                    machine.mode = Mode.TRACK
                    machine.track_goal = "default"
                    machine.transition = None

        step_envs(
            run,
            phase,
            frame,
            state,
            hand_state,
            odometry,
            machine.body_command,
            machine.hand_command,
            depth_diagnostics,
        )


def build_log(capacity, hand_dofs):
    log = recorder(capacity)
    log.phase = np.full(capacity, -1, dtype=np.int8)
    log.motion_frame = np.full(capacity, -1, dtype=np.int32)
    log.hand_pos = np.full((capacity, hand_dofs), np.nan, dtype=np.float32)
    log.hand_target = np.full((capacity, hand_dofs), np.nan, dtype=np.float32)
    log.depth_sequence = np.full(capacity, -1, dtype=np.int64)
    log.depth_age = np.full(capacity, np.nan, dtype=np.float32)
    log.depth_valid_ratio = np.full(capacity, np.nan, dtype=np.float32)
    log.depth_min = np.full(capacity, np.nan, dtype=np.float32)
    log.depth_max = np.full(capacity, np.nan, dtype=np.float32)
    log.phase_names = np.asarray(PHASES, dtype="<U16")
    return log


@hydra.main(version_base=None, config_path="../configs", config_name="run_loco_hoi_track")
def run(cfg: DictConfig) -> None:
    setup_logger()
    env = None
    hand_env = None
    camera = None
    depth_preview = None
    inspire_service = None
    log = None
    saved_terminal = None
    try:
        cfg.motion.name = select_motion(cfg.motion)
        loco_dof = compose_dof_config(cfg.robot.dof, cfg.loco.dof)
        track_dof = compose_dof_config(cfg.robot.dof, cfg.track.dof)
        loco = LeggedLabPolicy(cfg.loco, dof_cfg=loco_dof)
        hoi = BodyHandPolicy(
            cfg.hoi,
            cfg.motion,
            runtime_body_joint_names=cfg.robot.dof.joint_names,
            runtime_hand_joint_names=cfg.inspire.dof.joint_names,
            hand_mimic=cfg.inspire.mimic,
        )
        track = TrackPolicy(cfg.track, dof_cfg=track_dof)

        env = instantiate(cfg.env, dof_cfg=loco_dof, control_dt=loco.dt)
        if cfg.inspire_service:
            inspire_service = InspireService(
                cfg.env.net_if,
                cfg.env.domain_id,
                cfg.inspire_serial.left,
                cfg.inspire_serial.right,
            )
            inspire_service.start()
        hand_env = InspireHandEnv(dof_cfg=cfg.inspire.dof, domain_id=cfg.env.domain_id, net_if=cfg.env.net_if)
        controller = instantiate(cfg.controller, env=env)
        camera = instantiate(cfg.camera)

        body_control = cfg.hoi.action.body.control
        run = SimpleNamespace(
            env=env,
            hand_env=hand_env,
            controller=controller,
            loco=loco,
            hoi=hoi,
            track=track,
            camera=camera,
            depth_preview=None,
            depth_stale_seconds=float(cfg.hoi.depth.stale_seconds),
            dt=loco.dt,
            log=None,
            started=0.0,
            origin=0.0,
            key_descriptor=-1,
            hoi_stiffness=hoi.body_to_runtime.fit(body_control.stiffness),
            hoi_damping=hoi.body_to_runtime.fit(body_control.damping),
        )
        if cfg.recording.enabled:
            log = build_log(int(cfg.recording.seconds * loco.freq), hand_env.num_dofs)
            run.log = log

        env.self_check()
        hand_env.self_check()
        camera.self_check()
        if "depth_preview" in cfg:
            depth_preview = DepthPreviewServer(
                height=cfg.hoi.depth.height,
                width=cfg.hoi.depth.width,
                min_distance=cfg.hoi.depth.min_distance,
                max_distance=cfg.hoi.depth.max_distance,
                host=cfg.depth_preview.host,
                port=cfg.depth_preview.port,
            )
            run.depth_preview = depth_preview
        warm_up_policies(run)
        run.key_descriptor, saved_terminal = open_key_reader()

        body_command, hand_command = startup_locomotion(run, int(cfg.startup.ramp_seconds * loco.freq))
        run_pipeline(
            run,
            body_command,
            hand_command,
            int(cfg.handover.to_hoi_seconds * loco.freq),
            int(cfg.handover.to_default_seconds * loco.freq),
        )
    except Stop as stopped:
        logger.critical("Stopping: %s", stopped)
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        if saved_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved_terminal)
        if depth_preview is not None:
            depth_preview.shutdown()
        if camera is not None:
            camera.shutdown()
        if hand_env is not None:
            hand_env.shutdown()
        if env is not None:
            env.shutdown()
        if inspire_service is not None:
            inspire_service.stop()
        if log is not None:
            save_recording(log, cfg.recording.directory, OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    run()
