"""Closed-loop MuJoCo smoke test for the deployed body-hand policy.

Plays the staged reference motion clip end to end: the robot starts at the first
reference pose, the ONNX policy runs at 50 Hz, and PD torques are applied at
1 kHz. The run fails if the robot loses the reference or falls over.

    python scripts/sim_hoi_smoke.py
"""

import sys
from types import SimpleNamespace

import mujoco
import numpy as np
from omegaconf import OmegaConf

from g1_playground.inspire import dof as inspire_dof
from g1_playground.policy.body_hand import BodyHandPolicy
from g1_playground.policy.track import TrackPolicy
from g1_playground.simulation import G1MujocoBackend
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.dof import compose_dof_config
from g1_playground.utils.math import quat_angular_velocity, quat_inv, quat_mul, quat_slerp, yaw_quat

POLICY_CFG = "configs/policy/body_hand_distill_largebox.yaml"
MOTION_CFG = "configs/motion/largebox_039_v00.yaml"
ROBOT_CFG = "configs/robot/g1.yaml"
INSPIRE_CFG = "configs/robot/inspire.yaml"
TRACK_CFG = "configs/policy/track.yaml"
RUN_CFG = "configs/run_loco_largebox_track.yaml"
POLICY_DT = 0.02
PHYSICS_DT = 0.001
SUBSTEPS = int(POLICY_DT / PHYSICS_DT)
MAX_ANCHOR_POS_ERROR = 0.35
MIN_PELVIS_HEIGHT = 0.35
RETURN_HOLD_SECONDS = 2.0


def named_addresses(model, names):
    addresses = {}
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)  # pyright: ignore[reportAttributeAccessIssue]
        if joint < 0:
            raise RuntimeError(f"MuJoCo model has no joint named {name!r}")
        addresses[name] = int(model.jnt_qposadr[joint])
    return addresses


def read_joints(data, addresses, names):
    return np.asarray([data.qpos[addresses[name]] for name in names], dtype=np.float64), np.asarray(
        [data.qvel[addresses[name] - 1] for name in names], dtype=np.float64
    )  # qvel address is qpos address - 1 for 1-DoF hinge joints (free joint takes 7/6 slots up front)


def main() -> int:
    policy_cfg = OmegaConf.load(resolve_repo_path(POLICY_CFG))
    motion_cfg = OmegaConf.load(resolve_repo_path(MOTION_CFG))
    robot = OmegaConf.load(resolve_repo_path(ROBOT_CFG))
    inspire = OmegaConf.load(resolve_repo_path(INSPIRE_CFG))
    track_cfg = OmegaConf.load(resolve_repo_path(TRACK_CFG))
    handover_cfg = OmegaConf.load(resolve_repo_path(RUN_CFG)).handover
    max_terminal_linear_speed = float(handover_cfg.max_linear_speed)
    max_terminal_angular_speed = float(handover_cfg.max_angular_speed)
    max_terminal_joint_speed = float(handover_cfg.return_max_joint_speed)
    return_seconds = float(handover_cfg.to_standing_seconds)
    body_target_rate_limit = float(handover_cfg.body_rate_limit)

    policy = BodyHandPolicy(
        policy_cfg,
        motion_cfg,
        device="cpu",
        runtime_body_joint_names=robot.dof.joint_names,
        runtime_hand_joint_names=inspire.dof.joint_names,
        hand_mimic=inspire.mimic,
    )
    policy.motion.align()

    backend = G1MujocoBackend(
        resolve_repo_path(inspire.xml), PHYSICS_DT, elastic_support_scale=0.0, expected_actuators=53
    )
    model, data = backend.model, backend.data

    body_names = list(robot.dof.joint_names)
    hand_names = list(inspire.dof.joint_names)
    all_names = list(policy_cfg.observation.joint_names)
    addresses = named_addresses(model, all_names)

    # Initialize at the aligned first reference frame.
    data.qpos[0:3] = policy.motion.anchor_pos[0]
    data.qpos[3:7] = policy.motion.anchor_quat[0]
    for index, name in enumerate(all_names):
        data.qpos[addresses[name]] = policy.motion.joint_pos[0][index]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]

    # Per-actuator PD gains and torque limits, mapped by joint name.
    actuator_joints = inspire_dof.actuator_names(model)
    body_control = policy_cfg.action.body.control
    stiffness = dict(zip(policy_cfg.action.body.joint_names, body_control.stiffness, strict=True))
    damping = dict(zip(policy_cfg.action.body.joint_names, body_control.damping, strict=True))
    torque_limit = dict(zip(body_names, robot.dof.torque_limits, strict=True))
    kp = np.asarray([stiffness.get(name, inspire.sim.stiffness) for name in actuator_joints], dtype=np.float64)
    kd = np.asarray([damping.get(name, inspire.sim.damping) for name in actuator_joints], dtype=np.float64)
    limits = np.asarray(
        [torque_limit.get(name, inspire.sim.torque_limit) for name in actuator_joints], dtype=np.float64
    )

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")  # pyright: ignore[reportAttributeAccessIssue]
    policy.reset()
    pos_errors = []
    ori_errors = []
    for frame in range(policy.motion.num_frames):
        body_pos, body_vel = read_joints(data, addresses, body_names)
        hand_pos, hand_vel = read_joints(data, addresses, hand_names)
        body_state = SimpleNamespace(
            dof_pos=body_pos, dof_vel=body_vel, base_ang_vel=np.asarray(data.qvel[3:6], dtype=np.float32)
        )
        hand_state = SimpleNamespace(joint_pos=hand_pos, joint_vel=hand_vel)
        pelvis_quat = np.asarray(data.qpos[3:7], dtype=np.float32).copy()

        observation = policy.get_observation(frame, pelvis_quat, body_state, hand_state)
        body_target, hand_target = policy.act(observation)

        targets = dict(zip(body_names, body_target, strict=True))
        hand_targets = inspire_dof.to_dict(hand_target, hand_names)
        targets.update(inspire_dof.expand_mimic(hand_targets, inspire.mimic))

        for _ in range(SUBSTEPS):
            torque = np.empty(model.nu, dtype=np.float64)
            for index, name in enumerate(actuator_joints):
                address = addresses[name]
                torque[index] = kp[index] * (targets[name] - data.qpos[address]) - kd[index] * data.qvel[address - 1]
            backend.step(np.clip(torque, -limits, limits), support_scale=0.0)

        anchor_error = np.asarray(data.xpos[pelvis]) - policy.motion.anchor_pos[frame]
        pos_errors.append(float(np.linalg.norm(anchor_error)))
        delta = quat_mul(np.asarray(data.qpos[3:7], dtype=np.float32), quat_inv(policy.motion.anchor_quat[frame]))
        ori_errors.append(float(2.0 * np.arccos(np.clip(abs(float(delta[0])), -1.0, 1.0))))
        if data.qpos[2] < MIN_PELVIS_HEIGHT:
            print(f"FAIL: pelvis dropped to {data.qpos[2]:.3f} m at frame {frame}")
            return 1
        if pos_errors[-1] > MAX_ANCHOR_POS_ERROR:
            print(f"FAIL: anchor tracking error {pos_errors[-1]:.3f} m at frame {frame}")
            return 1

    _, terminal_joint_vel = read_joints(data, addresses, body_names)
    terminal_linear_speed = float(np.linalg.norm(data.qvel[:3]))
    terminal_angular_speed = float(np.linalg.norm(data.qvel[3:6]))
    terminal_joint_speed = float(np.max(np.abs(terminal_joint_vel)))
    terminal_height = float(data.qpos[2])
    if (
        terminal_linear_speed > max_terminal_linear_speed
        or terminal_angular_speed > max_terminal_angular_speed
        or terminal_joint_speed > max_terminal_joint_speed
    ):
        print(
            "FAIL: terminal reference is not settled: "
            f"linear {terminal_linear_speed:.3f} m/s, angular {terminal_angular_speed:.3f} rad/s, "
            f"max joint speed {terminal_joint_speed:.3f} rad/s"
        )
        return 1

    track_dof = compose_dof_config(robot.dof, track_cfg.dof)
    stand_track = TrackPolicy(track_cfg, device="cpu", dof_cfg=track_dof)
    stand_track.reset()
    stand_track.accept_applied_target(body_target)
    start_joint_pos, _ = read_joints(data, addresses, body_names)
    start_joint_pos = start_joint_pos.astype(np.float32)
    start_root_quat = np.asarray(data.qpos[3:7], dtype=np.float32).copy()
    target_root_quat = yaw_quat(start_root_quat)
    start_root_height = float(data.qpos[2])
    target_root_height = float(stand_track.reference_root_height)
    target_joint_pos = stand_track.standing_target.astype(np.float32)
    previous_anchor = stand_track.observation.anchor_pose(
        np.array([0.0, 0.0, start_root_height], dtype=np.float32), start_root_quat, start_joint_pos
    )

    return_frames = int(round(return_seconds / POLICY_DT))
    hold_frames = int(round(RETURN_HOLD_SECONDS / POLICY_DT))
    for return_frame in range(return_frames + hold_frames):
        if return_frame < return_frames:
            progress = (return_frame + 1) / return_frames
            alpha = progress * progress * (3.0 - 2.0 * progress)
            alpha_rate = 6.0 * progress * (1.0 - progress) / return_seconds
        else:
            alpha = 1.0
            alpha_rate = 0.0
        reference_height = (1.0 - alpha) * start_root_height + alpha * target_root_height
        reference_quat = quat_slerp(start_root_quat, target_root_quat, alpha)
        reference_joint_pos = (1.0 - alpha) * start_joint_pos + alpha * target_joint_pos
        reference_joint_vel = alpha_rate * (target_joint_pos - start_joint_pos)
        anchor = stand_track.observation.anchor_pose(
            np.array([0.0, 0.0, reference_height], dtype=np.float32),
            reference_quat,
            reference_joint_pos,
        )
        anchor_lin_vel = (anchor[0] - previous_anchor[0]) / POLICY_DT
        anchor_ang_vel = quat_angular_velocity(previous_anchor[1], anchor[1], POLICY_DT)
        stand_track.set_reference(
            root_height=reference_height,
            root_quat=reference_quat,
            joint_pos=reference_joint_pos,
            joint_vel=reference_joint_vel,
            anchor_lin_vel_w=anchor_lin_vel,
            anchor_ang_vel_w=anchor_ang_vel,
        )
        previous_anchor = anchor

        body_pos, body_vel = read_joints(data, addresses, body_names)
        body_state = SimpleNamespace(
            dof_pos=body_pos,
            dof_vel=body_vel,
            base_quat=np.asarray(data.qpos[3:7], dtype=np.float32)[[1, 2, 3, 0]],
            base_ang_vel=np.asarray(data.qvel[3:6], dtype=np.float32),
        )
        desired_body_target = stand_track.act(body_state, {"axes": {}})
        body_target = body_target + np.clip(
            desired_body_target - body_target, -body_target_rate_limit, body_target_rate_limit
        )
        stand_track.accept_applied_target(body_target)
        targets = dict(zip(body_names, body_target, strict=True))
        hand_targets = inspire_dof.to_dict(hand_target, hand_names)
        targets.update(inspire_dof.expand_mimic(hand_targets, inspire.mimic))
        for _ in range(SUBSTEPS):
            torque = np.empty(model.nu, dtype=np.float64)
            for index, name in enumerate(actuator_joints):
                address = addresses[name]
                torque[index] = kp[index] * (targets[name] - data.qpos[address]) - kd[index] * data.qvel[address - 1]
            backend.step(np.clip(torque, -limits, limits), support_scale=0.0)
        if data.qpos[2] < MIN_PELVIS_HEIGHT:
            print(f"FAIL: standing return fell to {data.qpos[2]:.3f} m at return frame {return_frame}")
            return 1

    _, return_joint_vel = read_joints(data, addresses, body_names)
    return_linear_speed = float(np.linalg.norm(data.qvel[:3]))
    return_angular_speed = float(np.linalg.norm(data.qvel[3:6]))
    return_joint_speed = float(np.max(np.abs(return_joint_vel)))
    if (
        return_linear_speed > max_terminal_linear_speed
        or return_angular_speed > max_terminal_angular_speed
        or return_joint_speed > max_terminal_joint_speed
    ):
        print(
            "FAIL: standing return did not settle: "
            f"linear {return_linear_speed:.3f} m/s, angular {return_angular_speed:.3f} rad/s, "
            f"max joint speed {return_joint_speed:.3f} rad/s"
        )
        return 1

    print(
        f"OK: {policy.motion.num_frames} frames, anchor pos error mean {np.mean(pos_errors):.4f} m "
        f"(max {np.max(pos_errors):.4f}), ori error mean {np.degrees(np.mean(ori_errors)):.2f} deg "
        f"(max {np.degrees(np.max(ori_errors)):.2f}), HOI terminal height {terminal_height:.3f} m, "
        f"terminal speed linear {terminal_linear_speed:.3f} m/s, angular {terminal_angular_speed:.3f} rad/s, "
        f"joint {terminal_joint_speed:.3f} rad/s; standing return height {data.qpos[2]:.3f} m, "
        f"speed linear {return_linear_speed:.3f} m/s, angular {return_angular_speed:.3f} rad/s, "
        f"joint {return_joint_speed:.3f} rad/s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
