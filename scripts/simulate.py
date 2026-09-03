import argparse
import importlib
import logging
import threading
import time

import mujoco
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from g1_playground.policy.body_hand.motion import aligned_object_frame_zero, select_motion
from g1_playground.simulation import G1MujocoBackend, G1MujocoDdsServer
from g1_playground.simulation.mujoco_viewer import MujocoViewer
from g1_playground.utils import resolve_repo_path

RENDER_HZ = 60.0
DEPTH_HZ = 60.0


def build_server(
    inspire: bool = False,
    *,
    object_mjcf: str | None = None,
    object_position=None,
    object_quaternion=None,
) -> G1MujocoDdsServer:
    robot = OmegaConf.load(resolve_repo_path("configs/robot/g1.yaml"))
    sim = OmegaConf.load(resolve_repo_path("configs/deployment/sim.yaml"))
    try:
        unitree_cpp = importlib.import_module("unitree_cpp")
    except ImportError as error:
        raise RuntimeError(
            "The MuJoCo DDS server requires the vendored unitree_cpp binding; "
            "run 'python scripts/setup/install_third_party.py unitree_cpp' first"
        ) from error

    endpoint = {
        "domain_id": sim.env.domain_id,
        "net_if": sim.env.net_if,
        "lowcmd_topic": sim.env.lowcmd_topic,
        "lowstate_topic": sim.env.lowstate_topic,
        "sport_state_topic": sim.env.sport_state_topic,
        "mode_machine": 5,
    }

    hand_cfg = None
    xml_path = robot.xml
    expected_actuators = 29
    if inspire:
        hand_cfg = OmegaConf.load(resolve_repo_path("configs/robot/inspire.yaml"))
        xml_path = hand_cfg.xml
        expected_actuators = 53

    backend_options = {
        "elastic_support_scale": 1.0,
        "expected_actuators": expected_actuators,
    }
    if object_mjcf is not None:
        backend_options.update(
            object_mjcf=resolve_repo_path(object_mjcf),
            object_position=object_position,
            object_quaternion=object_quaternion,
        )
    backend = G1MujocoBackend(resolve_repo_path(xml_path), **backend_options)

    body_index = None
    hand = None
    if inspire:
        from g1_playground.inspire import dof as inspire_dof
        from g1_playground.inspire.mujoco_hand import InspireMujocoHand

        names = inspire_dof.actuator_names(backend.model)
        body_index = [names.index(joint) for joint in robot.dof.joint_names]
        hand = InspireMujocoHand(
            backend.model,
            hand_cfg.dof,
            hand_cfg.mimic,
            domain_id=sim.env.domain_id,
            net_if=sim.env.net_if,
            stiffness=hand_cfg.sim.stiffness,
            damping=hand_cfg.sim.damping,
            torque_limit=hand_cfg.sim.torque_limit,
        )

    robot_endpoint = None
    try:
        robot_endpoint = unitree_cpp.G1DdsRobotEndpoint(endpoint)
        return G1MujocoDdsServer(
            backend,
            robot_endpoint,
            unitree_cpp.DdsLowStateSnapshot,
            robot.dof.torque_limits,
            body_index=body_index,
            hand=hand,
            sport_state_factory=unitree_cpp.DdsSportStateSnapshot,
            sport_publish_hz=float(sim.sim_odometry.publish_hz),
        )
    except BaseException:
        if robot_endpoint is not None:
            robot_endpoint.close()
        if hand is not None:
            hand.shutdown()
        raise


def run_with_viewer(server: G1MujocoDdsServer, *, depth: bool = False, depth_hz: float = DEPTH_HZ) -> None:
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []
    viewer = None
    worker = None
    started = False

    def run_server() -> None:
        try:
            server.run(stop_event)
        except BaseException as error:
            worker_errors.append(error)
        finally:
            stop_event.set()

    try:
        render_data = mujoco.MjData(server.backend.model)  # pyright: ignore[reportAttributeAccessIssue]
        server.backend.copy_data_to(render_data)
        viewer = MujocoViewer(server.backend.model, render_data, depth=depth)
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -10.0
        viewer.cam.azimuth = 180.0
        worker = threading.Thread(target=run_server, name="G1MujocoDdsServer")
        worker.start()
        started = True
        frame_period = 1.0 / RENDER_HZ
        frame_deadline = time.monotonic()
        depth_period = 1.0 / depth_hz
        depth_deadline = frame_deadline
        while viewer.is_running() and not stop_event.is_set():
            server.backend.copy_data_to(render_data)
            viewer.cam.lookat[:] = render_data.qpos[:3]
            now = time.monotonic()
            render_depth = depth and now >= depth_deadline
            if render_depth:
                depth_deadline += depth_period
                if depth_deadline < now:
                    depth_deadline = now + depth_period
            viewer.render(render_depth=render_depth, timestamp=now)
            frame_deadline += frame_period
            stop_event.wait(max(frame_deadline - time.monotonic(), 0.0))
    finally:
        stop_event.set()
        if started:
            if worker is not None:
                worker.join()
        else:
            server.shutdown()
        if viewer is not None:
            viewer.close()

    if worker_errors:
        raise worker_errors[0]


def run(argv=None) -> None:
    from g1_playground.utils.logger import setup_logger

    parser = argparse.ArgumentParser(description="G1 MuJoCo DDS simulator with a mandatory viewer")
    parser.add_argument(
        "--inspire",
        action="store_true",
        help="load the G1 model with Inspire hands and serve rt/inspire/cmd and rt/inspire/state",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="HOI Hydra overrides, for example hoi=depth/smalltable motion.name=sub17_smalltable_000_v02",
    )
    args = parser.parse_args(argv)

    setup_logger()
    try:
        server_options = {"inspire": args.inspire}
        viewer_options = {"depth": args.inspire}
        if args.overrides:
            if not args.inspire:
                parser.error("HOI scene overrides require --inspire")
            with initialize_config_dir(version_base=None, config_dir=resolve_repo_path("configs")):
                cfg = compose(config_name="run_loco_hoi_track", overrides=args.overrides)
            cfg.motion.name = select_motion(cfg.motion)
            object_position, object_quaternion = aligned_object_frame_zero(cfg.motion)
            server_options.update(
                object_mjcf=str(cfg.hoi.object_mjcf),
                object_position=object_position,
                object_quaternion=object_quaternion,
            )
            logging.getLogger("g1_playground").warning(
                "Initialized %s for motion %s at position %s, quaternion wxyz %s",
                cfg.hoi.object,
                cfg.motion.name,
                object_position,
                object_quaternion,
            )
        server = build_server(**server_options)
        run_with_viewer(server, **viewer_options)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    run()
