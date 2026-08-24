import argparse
import os

import numpy as np
from omegaconf import OmegaConf

DEFAULT_POLICY = "configs/policy/body_hand_distill_largebox.yaml"
DEFAULT_SOURCE = "hoi/data/train/track_H20_var5_1perclip.npz"
DEFAULT_TARGET = "assets/motions/largebox"
ANCHOR_INDEX = 0


def repo_path(value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), value))


def locate_clip(data, clip: str) -> tuple[int, int, int, str]:
    names = [str(name) for name in data["motion_names"]]
    lengths = np.asarray(data["motion_lengths"], dtype=np.int64)
    if clip.isdigit():
        index = int(clip)
        if not 0 <= index < len(names):
            raise SystemExit(f"Clip index {index} is outside the {len(names)}-clip motion file")
    else:
        if clip not in names:
            raise SystemExit(f"{clip} is not in the motion file; first names are {names[:3]}")
        index = names.index(clip)
    return index, int(lengths[:index].sum()), int(lengths[index]), names[index]


def build_motion(source: str, clip: str, joint_names) -> dict:
    data = np.load(source, allow_pickle=True)
    index, start, length, clip_name = locate_clip(data, clip)
    rows = slice(start, start + length)

    joint_pos = np.asarray(data["joint_pos"][rows], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel"][rows], dtype=np.float32)
    anchor_pos = np.asarray(data["body_pos_w"][rows, ANCHOR_INDEX], dtype=np.float32)
    anchor_quat = np.asarray(data["body_quat_w"][rows, ANCHOR_INDEX], dtype=np.float32)
    anchor_lin_vel = np.asarray(data["body_lin_vel_w"][rows, ANCHOR_INDEX], dtype=np.float32)

    if joint_pos.shape != (length, len(joint_names)) or joint_vel.shape != joint_pos.shape:
        raise SystemExit(
            f"Motion stores {joint_pos.shape[-1]} joints but deployment declares {len(joint_names)}"
        )
    norms = np.linalg.norm(anchor_quat.astype(np.float64), axis=1)
    deviation = float(np.abs(norms - 1.0).max())
    if deviation > 1e-3:
        raise SystemExit(f"Anchor quaternion is not normalized: max |q| deviation {deviation:.6f}")
    for name, values in (
        ("joint_pos", joint_pos),
        ("joint_vel", joint_vel),
        ("anchor_pos_w", anchor_pos),
        ("anchor_quat_w", anchor_quat),
        ("anchor_lin_vel_w", anchor_lin_vel),
    ):
        if not np.all(np.isfinite(values)):
            raise SystemExit(f"Clip {clip_name} holds non-finite {name}")
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "anchor_pos_w": anchor_pos,
        "anchor_quat_w": anchor_quat,
        "anchor_lin_vel_w": anchor_lin_vel,
        "joint_names": np.asarray(joint_names, dtype="<U32"),
        "fps": np.asarray([fps], dtype=np.int64),
        "clip_name": np.asarray(clip_name),
        "clip_index": np.asarray([index], dtype=np.int64),
        "source_file": np.asarray(f"{os.path.basename(source)}#{clip_name}"),
    }


def run(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract one packed HOI training clip into the deployment motion format"
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="the packed training motion .npz")
    parser.add_argument("--clip", default="0", help="clip index or clip name")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="deployment policy config for the joint order")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--out-name", default=None, help="output file name; defaults to the clip name")
    args = parser.parse_args(argv)

    policy = OmegaConf.load(repo_path(args.policy))
    joint_names = [str(name) for name in policy.observation.joint_names]
    motion = build_motion(repo_path(args.source), args.clip, joint_names)

    fps = int(motion["fps"][0])
    if fps != int(policy.frequency):
        raise SystemExit(f"Motion is {fps} Hz but the policy runs at {policy.frequency} Hz")

    target = repo_path(args.target)
    os.makedirs(target, exist_ok=True)
    name = args.out_name or str(motion["clip_name"])
    path = os.path.join(target, f"{name}.npz")
    np.savez_compressed(path, **motion)

    frames = int(motion["joint_pos"].shape[0])
    speed = float(np.linalg.norm(motion["anchor_lin_vel_w"], axis=1).mean())
    print(f"Extracted {motion['clip_name']}: {frames} frames ({frames / fps:.2f} s) at {fps} Hz")
    print(f"  pelvis height {float(motion['anchor_pos_w'][:, 2].mean()):.4f} m, mean speed {speed:.3f} m/s")
    print(f"Wrote {path}")


if __name__ == "__main__":
    run()
