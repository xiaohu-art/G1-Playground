import logging
import os
import select
import sys
import termios
import tty

import numpy as np

from g1_playground.utils import resolve_repo_path
from g1_playground.utils.math import TransformAlignment

logger = logging.getLogger("g1_playground")
MOTION_UP_KEY = b"\x1b[A"
MOTION_DOWN_KEY = b"\x1b[B"


def motion_bounds(motions, name, path) -> tuple[int, int]:
    names = [str(value) for value in motions["motion_names"]]
    try:
        index = names.index(str(name))
    except ValueError as error:
        raise ValueError(f"Reference motion {name!r} is not in {path}") from error
    lengths = np.asarray(motions["motion_lengths"], dtype=np.int64)
    start = int(lengths[:index].sum())
    return start, start + int(lengths[index])


def select_motion(cfg_motion) -> str:
    with np.load(resolve_repo_path(cfg_motion.file), allow_pickle=False) as motions:
        names = [str(name) for name in motions["motion_names"]]
    selected = str(cfg_motion.name)
    if selected not in names:
        selected = names[0]
    if not cfg_motion.get("interactive", True):
        logger.warning("Motion selection is non-interactive; using configured HOI motion %s", selected)
        return selected
    if not sys.stdin.isatty():
        logger.warning("stdin is not a terminal; using configured HOI motion %s", selected)
        return selected

    descriptor = sys.stdin.fileno()
    saved = termios.tcgetattr(descriptor)
    index = names.index(selected)
    try:
        tty.setraw(descriptor)
        while True:
            sys.stdout.write(
                f"\r\033[2KSelect HOI motion with Up/Down, Enter confirms [{index + 1}/{len(names)}]: {names[index]}"
            )
            sys.stdout.flush()
            key = os.read(descriptor, 1)
            if key == b"\x1b":
                for _ in range(2):
                    if not select.select([descriptor], [], [], 0.05)[0]:
                        break
                    key += os.read(descriptor, 1)
            if key == MOTION_UP_KEY:
                index = (index - 1) % len(names)
            elif key == MOTION_DOWN_KEY:
                index = (index + 1) % len(names)
            elif key in (b"\r", b"\n"):
                selected = names[index]
                break
            elif key == b"\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
        sys.stdout.write("\n")
        sys.stdout.flush()
    logger.warning("Selected HOI motion: %s", selected)
    return selected


def aligned_object_frame_zero(cfg_motion) -> tuple[np.ndarray, np.ndarray]:
    with np.load(resolve_repo_path(cfg_motion.file), allow_pickle=False) as motions:
        start, _ = motion_bounds(motions, cfg_motion.name, cfg_motion.file)
        alignment = TransformAlignment(
            quat=motions["anchor_quat_w"][start],
            pos=motions["anchor_pos_w"][start],
            yaw_only=True,
            xy_only=True,
        )
        position = alignment.align_pos(motions["object_pos_w"][start])
        quaternion = alignment.align_quat(motions["object_quat_w"][start])
    return position, quaternion
