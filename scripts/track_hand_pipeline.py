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
import viser
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from g1_playground.inspire import dof as inspire_dof
from g1_playground.inspire.hand_env import InspireHandEnv
from g1_playground.inspire.service import InspireService
from g1_playground.policy.track import TrackPolicy
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.logger import setup_logger
from g1_playground.utils.math import is_upright

logger = logging.getLogger("g1_playground")
RAMP_SECONDS = 3.0
BLEND_SECONDS = 5.0
IDLE_CONTROL = {"axes": {}}
VISER_PORT = 8080


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


def build_hand_sliders(hand_env, joint_names, port):
    server = viser.ViserServer(port=port, label="G1 track policy + Inspire hand")
    server.gui.add_markdown(
        "The body is held by the track policy. Move a slider and watch that finger. "
        "Values are radians; the lower limit is fully open."
    )
    readback = server.gui.add_text("readback", initial_value="waiting", disabled=True)
    sliders = []
    for index, name in enumerate(joint_names):
        low = float(hand_env.lower[index])
        high = float(hand_env.upper[index])
        sliders.append(
            server.gui.add_slider(
                f"{index:2d} {name}",
                min=low,
                max=high,
                step=(high - low) / 200.0,
                initial_value=low,
            )
        )

    open_button = server.gui.add_button("open all")
    close_button = server.gui.add_button("close all")

    @open_button.on_click
    def _(_):
        for index, slider in enumerate(sliders):
            slider.value = float(hand_env.lower[index])

    @close_button.on_click
    def _(_):
        for index, slider in enumerate(sliders):
            slider.value = float(hand_env.upper[index])

    logger.warning("viser hand sliders on http://localhost:%d", port)
    return sliders, readback


@hydra.main(version_base=None, config_path="../configs", config_name="run_track")
def run(cfg: DictConfig) -> None:
    setup_logger()
    env = None
    hand = None
    service = None
    try:
        dof = compose_dof_config(cfg.robot.dof, cfg.policy.dof)
        policy = TrackPolicy(cfg.policy, dof_cfg=dof)
        env = instantiate(cfg.env, dof_cfg=dof, control_dt=policy.dt)
        controller = instantiate(cfg.controller, env=env)

        inspire = OmegaConf.load(resolve_repo_path("configs/robot/inspire.yaml"))
        if cfg.inspire_service:
            service = InspireService(
                cfg.env.net_if,
                cfg.env.domain_id,
                cfg.inspire_serial.left,
                cfg.inspire_serial.right,
            )
            service.start()
        hand = InspireHandEnv(
            dof_cfg=inspire.dof,
            domain_id=cfg.env.domain_id,
            net_if=cfg.env.net_if,
        )
        hand_names = inspire_dof.joint_names(inspire.dof)

        env.self_check()
        hand.self_check()
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

        sliders, readback = build_hand_sliders(hand, hand_names, VISER_PORT)
        logger.warning("Hand control is live; the body stays under the track policy")

        tick = 0
        while True:
            started = time.monotonic()
            if not step(env, controller, policy):
                break
            hand.step([slider.value for slider in sliders])

            tick += 1
            if tick % 10 == 0:
                hand_state = hand.read()
                flag = "STALE" if hand_state.stale else f"{hand_state.age * 1000:.0f}ms"
                readback.value = " ".join(f"{v:+.2f}" for v in hand_state.joint_pos) + f"   [{flag}]"

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
        if hand is not None:
            try:
                for _ in range(int(0.5 / 0.02)):
                    hand.step(hand.lower)
                    time.sleep(0.02)
            except Exception as error:
                logger.error("Failed to open the hand: %r", error)
            hand.shutdown()
        if env is not None:
            env.shutdown()
        if service is not None:
            service.stop()


if __name__ == "__main__":
    run()
