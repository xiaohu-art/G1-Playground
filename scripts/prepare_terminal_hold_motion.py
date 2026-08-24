import argparse
import os

import numpy as np
from omegaconf import OmegaConf

DEFAULT_POLICY = "configs/policy/body_hand_distill_largebox.yaml"
DEFAULT_TARGET = "assets/motions/largebox"
ANCHOR_INDEX = 0


def repo_path(value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), value))


def locate_clip(data, clip_name: str) -> tuple[int, int]:
    names = [str(name) for name in data["motion_names"]]
    if clip_name not in names:
        raise SystemExit(f"{clip_name} is not in the motion file; first names are {names[:3]}")
    lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
    index = names.index(clip_name)
    return int(lengths[:index].sum()), int(lengths[index])


def build_hold(motion_file: str, clip_name: str, frame: int | None, frames: int, joint_names) -> dict:
    data = np.load(motion_file, allow_pickle=True)
    start, length = locate_clip(data, clip_name)
    local = length - 1 if frame is None else int(frame)
    if not 0 <= local < length:
        raise SystemExit(f"Frame {local} is outside the {length}-frame clip {clip_name}")
    row = start + local

    joint_pos = np.asarray(data["joint_pos"][row], dtype=np.float32)
    if joint_pos.shape != (len(joint_names),):
        raise SystemExit(f"Motion stores {joint_pos.shape[0]} joints but deployment declares {len(joint_names)}")
    anchor_pos = np.asarray(data["body_pos_w"][row, ANCHOR_INDEX], dtype=np.float32)
    anchor_quat = np.asarray(data["body_quat_w"][row, ANCHOR_INDEX], dtype=np.float32)

    norm = float(np.linalg.norm(anchor_quat))
    if abs(norm - 1.0) > 1e-3:
        raise SystemExit(f"Anchor quaternion is not normalized: |q| = {norm:.6f}")
    for name, values in (("joint_pos", joint_pos), ("anchor_pos_w", anchor_pos), ("anchor_quat_w", anchor_quat)):
        if not np.all(np.isfinite(values)):
            raise SystemExit(f"Source frame holds non-finite {name}")
    contacts = np.asarray(data["contact_label"][row])
    if int((contacts > 0).sum()):
        raise SystemExit(f"Frame {local} of {clip_name} has {int((contacts > 0).sum())} contact labels; pick another")
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    return (
        {
            "joint_pos": np.repeat(joint_pos[None], frames, axis=0),
            "joint_vel": np.zeros((frames, len(joint_names)), dtype=np.float32),
            "anchor_pos_w": np.repeat(anchor_pos[None], frames, axis=0),
            "anchor_quat_w": np.repeat(anchor_quat[None], frames, axis=0),
            "anchor_lin_vel_w": np.zeros((frames, 3), dtype=np.float32),
            "joint_names": np.asarray(joint_names, dtype="<U32"),
            "fps": np.asarray([fps], dtype=np.int64),
            "clip_name": np.asarray(f"{clip_name}_terminal_hold"),
            "source_file": np.asarray(f"{os.path.basename(motion_file)}#{clip_name}@{local}"),
        },
        local,
        length,
    )


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Freeze one reference frame into a quasi-static hold motion")
    parser.add_argument("--motion", required=True, help="the source motion .npz the frame is taken from")
    parser.add_argument("--clip-name", required=True)
    parser.add_argument("--frame", type=int, default=None, help="local frame index; defaults to the clip's last")
    parser.add_argument("--frames", type=int, default=100, help="how many identical frames to emit")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="deployment policy config for the joint order")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args(argv)

    if args.frames < 1:
        raise SystemExit("A hold needs at least one frame")
    policy = OmegaConf.load(repo_path(args.policy))
    joint_names = [str(name) for name in policy.observation.joint_names]
    hold, local, length = build_hold(args.motion, args.clip_name, args.frame, args.frames, joint_names)

    fps = int(hold["fps"][0])
    if fps != int(policy.frequency):
        raise SystemExit(f"Motion is {fps} Hz but the policy runs at {policy.frequency} Hz")
    for key in ("joint_pos", "anchor_pos_w", "anchor_quat_w"):
        if not np.all(hold[key] == hold[key][0]):
            raise SystemExit(f"{key} is not identical across the hold")

    target = repo_path(args.target)
    os.makedirs(target, exist_ok=True)
    path = os.path.join(target, f"{hold['clip_name']}.npz")
    np.savez_compressed(path, **hold)

    quaternion = hold["anchor_quat_w"][0].astype(np.float64)
    w, x, y, z = quaternion
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))
    print(f"Held {args.clip_name} frame {local} of {length - 1} for {args.frames} frames ({args.frames / fps:.2f} s)")
    print(f"  pelvis height {hold['anchor_pos_w'][0][2]:.4f} m, tilt {tilt:.2f} deg, no contact labels")
    print(f"  {len(joint_names)} joints in the deployment observation order")
    print(f"Wrote {path}")


if __name__ == "__main__":
    run()
