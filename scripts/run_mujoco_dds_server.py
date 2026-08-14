import importlib
import threading
import time

import mujoco
from omegaconf import OmegaConf

from g1_playground.simulation import G1MujocoBackend, G1MujocoDdsServer
from g1_playground.utils import resolve_repo_path

RENDER_HZ = 60.0


def build_server() -> G1MujocoDdsServer:
    robot = OmegaConf.load(resolve_repo_path("configs/robot/g1.yaml"))
    sim = OmegaConf.load(resolve_repo_path("configs/deployment/sim.yaml"))
    try:
        unitree_cpp = importlib.import_module("unitree_cpp")
    except ImportError as error:
        raise RuntimeError(
            "The MuJoCo DDS server requires the vendored unitree_cpp binding; "
            "run 'python scripts/install_third_party.py unitree_cpp' first"
        ) from error

    endpoint = {
        "domain_id": sim.env.domain_id,
        "net_if": sim.env.net_if,
        "lowcmd_topic": sim.env.lowcmd_topic,
        "lowstate_topic": sim.env.lowstate_topic,
        "mode_machine": 5,
    }

    backend = G1MujocoBackend(
        resolve_repo_path(robot.xml),
        elastic_support_scale=1.0,
    )
    bridge = None
    try:
        bridge = unitree_cpp.G1DdsSimServer(endpoint)
        return G1MujocoDdsServer(
            backend,
            bridge,
            unitree_cpp.DdsLowStateSnapshot,
            robot.dof.torque_limits,
        )
    except BaseException:
        if bridge is not None:
            bridge.close()
        raise


def run_with_viewer(server: G1MujocoDdsServer) -> None:
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
        viewer_module = importlib.import_module("mujoco.viewer")
        render_data = mujoco.MjData(server.backend.model)  # pyright: ignore[reportAttributeAccessIssue]
        server.backend.copy_data_to(render_data)
        viewer = viewer_module.launch_passive(
            server.backend.model,
            render_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        with viewer.lock():
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -10.0
            viewer.cam.azimuth = 180.0

        worker = threading.Thread(target=run_server, name="G1MujocoDdsServer")
        worker.start()
        started = True
        frame_period = 1.0 / RENDER_HZ
        frame_deadline = time.monotonic()
        while viewer.is_running() and not stop_event.is_set():
            with viewer.lock():
                server.backend.copy_data_to(render_data)
                viewer.cam.lookat[:] = render_data.qpos[:3]
            viewer.sync()
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


def run() -> None:
    from g1_playground.utils.logger import setup_logger

    setup_logger()
    try:
        server = build_server()
        run_with_viewer(server)
    except KeyboardInterrupt:
        return


def main() -> None:
    run()


if __name__ == "__main__":
    main()
