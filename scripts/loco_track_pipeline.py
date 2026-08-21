# ruff: noqa: I001

import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import logging
import select
import sys
import termios
import time
import tty
from types import SimpleNamespace

# Run the Jetson torch-before-numpy bootstrap before anything that can import NumPy.
import g1_playground  # noqa: F401

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from g1_playground.policy import UnitreeWoGaitPolicy
from g1_playground.policy.track import TrackPolicy
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import is_upright

logger = logging.getLogger("g1_playground")
RAMP_SECONDS = 3.0
BLEND_SECONDS = 5.0
DRY_FRAMES = 10
FRAME_DROP_LIMIT = 0.2
SWITCH_KEY = b"]"
ZERO_CONTROL = {"axes": {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0}}
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


def pace(started: float, period: float) -> float:
    remaining = period - (time.monotonic() - started)
    if remaining > 0:
        time.sleep(remaining)
    return remaining


def eased(progress: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * min(max(progress, 0.0), 1.0)))


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


def commands(env, loco, track, plan, measured, standing):
    state, control, requested = yield

    logger.warning("Ramping to the locomotion standing pose over %.1f seconds", RAMP_SECONDS)
    for index in range(plan.ramp_steps):
        alpha = (index + 1) / plan.ramp_steps
        state, control, requested = yield (1.0 - alpha) * measured + alpha * standing

    loco.reset()

    logger.warning("Blending into closed-loop locomotion over %.1f seconds", BLEND_SECONDS)
    command = standing
    for index in range(plan.blend_steps):
        alpha = (index + 1) / plan.blend_steps
        command = (1.0 - alpha) * standing + alpha * loco.act(state, ZERO_CONTROL)
        state, control, requested = yield command

    while not requested:
        command = loco.act(state, control)
        state, control, requested = yield command

    logger.warning("Handing over to the tracking policy over %.1f seconds", plan.handover_seconds)
    track.reset()
    for index in range(plan.handover_steps):
        alpha = eased((index + 1) / plan.handover_steps)
        blended = (1.0 - alpha) * loco.act(state, ZERO_CONTROL) + alpha * track.act(state, IDLE_CONTROL)
        command = command + np.clip(blended - command, -plan.rate_limit, plan.rate_limit)
        state, control, requested = yield command

    env.set_gains(plan.track_stiffness, plan.track_damping)
    logger.warning("Tracking policy is now in control")

    while True:
        state, control, requested = yield track.act(state, IDLE_CONTROL)


@hydra.main(version_base=None, config_path="../configs", config_name="run_loco_track")
def run(cfg: DictConfig) -> None:
    setup_logger()
    torch.set_num_threads(1)
    env = None
    saved_terminal = None
    try:
        dof_loco = compose_dof_config(cfg.robot.dof, cfg.loco.dof)
        dof_track = compose_dof_config(cfg.robot.dof, cfg.track.dof)
        loco = UnitreeWoGaitPolicy(cfg.loco, device=cfg.device, dof_cfg=dof_loco)
        track = TrackPolicy(cfg.track, device=cfg.device, dof_cfg=dof_track)
        env = instantiate(cfg.env, dof_cfg=dof_loco, control_dt=loco.dt)
        controller = instantiate(cfg.controller, env=env)
        plan = SimpleNamespace(
            ramp_steps=int(RAMP_SECONDS * loco.freq),
            blend_steps=int(BLEND_SECONDS * loco.freq),
            handover_seconds=float(cfg.handover.crossfade_seconds),
            handover_steps=int(cfg.handover.crossfade_seconds * loco.freq),
            rate_limit=float(cfg.handover.rate_limit),
            track_stiffness=np.asarray(dof_track.stiffness),
            track_damping=np.asarray(dof_track.damping),
        )

        env.self_check()
        for _ in range(DRY_FRAMES):
            frame = read_frame(env, controller)
            if frame is None:
                return
            state, control = frame
            loco.act(state, control)
            track.act(state, IDLE_CONTROL)

        frame = read_frame(env, controller)
        if frame is None:
            return
        state, _ = frame

        key_descriptor, saved_terminal = open_key_reader()
        frames = commands(env, loco, track, plan, state.dof_pos, loco.standing_target)
        next(frames)
        env.activate_commands()

        while True:
            started = time.monotonic()
            frame = read_frame(env, controller)
            if frame is None:
                break
            state, control = frame
            requested, key_descriptor = poll_switch_key(key_descriptor)
            env.step(frames.send((state, control, requested)))
            if pace(started, loco.dt) < -FRAME_DROP_LIMIT:
                logger.critical("Exiting due to excessive frame drop")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by operator")
    finally:
        if saved_terminal is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved_terminal)
        if env is not None:
            env.shutdown()


if __name__ == "__main__":
    run()
