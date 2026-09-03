import logging

import numpy as np
from omegaconf import DictConfig

from g1_playground.policy.body_hand.depth import DepthBodyHandObservation
from g1_playground.policy.body_hand.motion import motion_bounds
from g1_playground.policy.body_hand.observation import JointAssembler, ReferenceMotion
from g1_playground.policy.tensorrt_runner import TensorRTRunner
from g1_playground.utils import resolve_repo_path
from g1_playground.utils.dof import DoFAdapter

logger = logging.getLogger(__name__)


class BodyHandPolicy:
    def __init__(
        self,
        cfg_policy: DictConfig,
        cfg_motion: DictConfig,
        *,
        runtime_body_joint_names,
        runtime_hand_joint_names,
        hand_mimic,
        runner=None,
    ):
        self.runner = runner or TensorRTRunner(resolve_repo_path(cfg_policy.policy_file))
        if self.runner.input_names != ("obs",) or self.runner.output_names != ("actions",):
            raise ValueError(
                f"Body-hand policy requires obs -> actions, got {self.runner.input_names} -> {self.runner.output_names}"
            )

        self.freq = int(cfg_policy.frequency)
        self.dt = 1.0 / self.freq
        self.observation_dim = self.runner.shape("obs")[-1]
        self.action_dim = self.runner.shape("actions")[-1]

        observation_joint_names = list(cfg_policy.observation.joint_names)
        body_action_names = list(cfg_policy.action.body.joint_names)
        hand_action_names = list(cfg_policy.action.hand.joint_names)
        runtime_body_joint_names = tuple(runtime_body_joint_names)
        runtime_hand_joint_names = tuple(runtime_hand_joint_names)

        self.assembler = JointAssembler(
            observation_joint_names, runtime_body_joint_names, runtime_hand_joint_names, hand_mimic
        )
        self.body_to_runtime = DoFAdapter(body_action_names, runtime_body_joint_names)
        self.hand_to_runtime = DoFAdapter(hand_action_names, runtime_hand_joint_names)
        self.body_action_dim = len(body_action_names)

        self.action_scale = np.asarray(cfg_policy.action.scale, dtype=np.float32)
        self.action_offset = np.asarray(cfg_policy.action.offset, dtype=np.float32)
        default_joint_pos = np.asarray(cfg_policy.observation.default_joint_pos, dtype=np.float32)
        if (
            len(body_action_names) + len(hand_action_names) != self.action_dim
            or self.action_scale.shape != (self.action_dim,)
            or self.action_offset.shape != (self.action_dim,)
            or default_joint_pos.shape != (len(observation_joint_names),)
        ):
            raise ValueError("Body-hand policy configuration does not match the ONNX tensor dimensions")

        self.motion = self._load_motion(cfg_motion, observation_joint_names, cfg_policy.observation.future_offsets)
        self.observation = DepthBodyHandObservation(
            self.motion,
            default_joint_pos,
            self.observation_dim,
            height=cfg_policy.depth.height,
            width=cfg_policy.depth.width,
            min_distance=cfg_policy.depth.min_distance,
            max_distance=cfg_policy.depth.max_distance,
        )

        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        initial_observation = (
            0,
            self.motion.anchor_pos[0],
            self.motion.anchor_quat[0],
            np.zeros(3, dtype=np.float32),
            default_joint_pos,
            np.zeros_like(default_joint_pos),
            self._last_action,
        )
        self.observation.build(
            *initial_observation,
            np.zeros((cfg_policy.depth.height, cfg_policy.depth.width), dtype=np.float32),
        )

        logger.info(
            "Body-hand policy: observation [%d] -> action [%d] at %d Hz",
            self.observation_dim,
            self.action_dim,
            self.freq,
        )

    def _load_motion(self, cfg_motion, observation_joint_names, future_offsets) -> ReferenceMotion:
        with np.load(resolve_repo_path(cfg_motion.file), allow_pickle=False) as motions:
            if [str(name) for name in motions["joint_names"]] != observation_joint_names:
                raise ValueError("Reference motion joint names do not match the policy configuration")
            fps = int(np.asarray(motions["fps"]).reshape(-1)[0])
            if fps != self.freq:
                raise ValueError(f"Reference motion is {fps} Hz but the policy runs at {self.freq} Hz")

            start, stop = motion_bounds(motions, cfg_motion.name, cfg_motion.file)
            return ReferenceMotion(
                motions["joint_pos"][start:stop],
                motions["anchor_pos_w"][start:stop],
                motions["anchor_quat_w"][start:stop],
                motions["object_pos_w"][start:stop],
                motions["object_quat_w"][start:stop],
                motions["contact_label"][start:stop],
                future_offsets,
            )

    def infer(self, observation: np.ndarray) -> np.ndarray:
        outputs = self.runner.run({"obs": observation.reshape(1, -1)})
        return np.asarray(outputs["actions"], dtype=np.float32).reshape(-1)

    def process(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(self.action_dim)
        return raw_action * self.action_scale + self.action_offset

    def split(self, processed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        processed = np.asarray(processed, dtype=np.float32).reshape(self.action_dim)
        body_action = processed[: self.body_action_dim]
        hand_action = processed[self.body_action_dim :]
        return self.body_to_runtime.fit(body_action).copy(), self.hand_to_runtime.fit(hand_action).copy()

    def joint_state(self, body_state, hand_state) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.assembler.positions(body_state.dof_pos, hand_state.joint_pos),
            self.assembler.velocities(body_state.dof_vel, hand_state.joint_vel),
        )

    def reference_targets(self) -> tuple[np.ndarray, np.ndarray]:
        joint_pos = self.motion.joint_pos[0]
        return self.assembler.state_to_body.fit(joint_pos), self.assembler.state_to_hand.fit(joint_pos)

    def reset(self) -> None:
        self._last_action.fill(0.0)

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def get_observation(self, frame, anchor_pos, anchor_quat, body_state, hand_state, *, depth_m) -> np.ndarray:
        joint_pos, joint_vel = self.joint_state(body_state, hand_state)
        arguments = (
            frame,
            anchor_pos,
            anchor_quat,
            body_state.base_ang_vel,
            joint_pos,
            joint_vel,
            self._last_action,
        )
        return self.observation.build(*arguments, depth_m)

    def act(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw_action = self.infer(observation)
        if not np.all(np.isfinite(raw_action)):
            raise RuntimeError("Body-hand policy emitted a non-finite action")
        self._last_action = raw_action.copy()
        return self.split(self.process(raw_action))
