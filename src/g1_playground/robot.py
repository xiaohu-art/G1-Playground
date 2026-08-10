import time
import threading
import numpy as np

from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__HandCmd_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    LowCmd_,
    LowState_,
    HandCmd_,
    HandState_,
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from g1_playground.state import G1State
from g1_playground.policies.policy import Policy
from g1_playground.action import JointAction


LEGS_Kp = [60, 60, 60, 100, 40, 40]
WAIST_Kp = [60, 40, 40]
ARMS_Kp = [40, 40, 40, 40, 40, 40, 40]
DEFAULT_KP = np.array(LEGS_Kp + LEGS_Kp + WAIST_Kp + ARMS_Kp + ARMS_Kp, dtype=np.float32)

LEGS_Kd = [1, 1, 1, 2, 1, 1]
WAIST_Kd = [1, 1, 1]
ARMS_Kd = [1, 1, 1, 1, 1, 1, 1]
DEFAULT_KD = np.array(LEGS_Kd + LEGS_Kd + WAIST_Kd + ARMS_Kd + ARMS_Kd, dtype=np.float32)

BODY_NUM_MOTORS = 29
HAND_NUM_MOTORS = 7

G1_29DOF_NUM_MOTORS = BODY_NUM_MOTORS
G1_29DOF_DEX3_NUM_MOTORS = BODY_NUM_MOTORS + 2 * HAND_NUM_MOTORS  # 43
VALID_NUM_MOTORS = {G1_29DOF_NUM_MOTORS, G1_29DOF_DEX3_NUM_MOTORS}

DEFAULT_HAND_KP = np.full(HAND_NUM_MOTORS, 1.5, dtype=np.float32)
DEFAULT_HAND_KD = np.full(HAND_NUM_MOTORS, 0.1, dtype=np.float32)

BODY_SLICE = slice(0, BODY_NUM_MOTORS)
LEFT_HAND_SLICE = slice(BODY_NUM_MOTORS, BODY_NUM_MOTORS + HAND_NUM_MOTORS)
RIGHT_HAND_SLICE = slice(BODY_NUM_MOTORS + HAND_NUM_MOTORS, G1_29DOF_DEX3_NUM_MOTORS)


class G129DofJointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13
    WaistA = 13
    WaistPitch = 14
    WaistB = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28


class G1Dex3HandJointIndex:
    Thumb0 = 0
    Thumb1 = 1
    Thumb2 = 2
    Middle0 = 3
    Middle1 = 4
    Index0 = 5
    Index1 = 6

class UnitreeG1Robot:
    def __init__(
        self,
        policy: Policy,
        control_dt: float = 0.002,
        num_motors: int = G1_29DOF_NUM_MOTORS,
    ):
        if num_motors not in VALID_NUM_MOTORS:
            raise ValueError(
                f"num_motors must be {G1_29DOF_NUM_MOTORS} (body only) or "
                f"{G1_29DOF_DEX3_NUM_MOTORS} (body + 2 Dex3 hands), got {num_motors}."
            )

        self.policy = policy
        self.control_dt = control_dt
        self.num_motors = num_motors
        self.has_hands = num_motors == G1_29DOF_DEX3_NUM_MOTORS

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state: LowState_ | None = None
        self.mode_machine: int | None = None
        self.updated_mode_machine = False

        if self.has_hands:
            self.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.left_hand_state: HandState_ | None = None
            self.right_hand_state: HandState_ | None = None

        self._target_action: JointAction | None = None
        self._target_lock = threading.Lock()
        self.crc = CRC()

    def initialize(self):
        self._release_motion_mode()
        self._create_publishers()
        self._create_subscribers()

    def _release_motion_mode(self):
        """Release any active high-level motion service on the robot.

        The G1 ships with built-in motion services (sport_mode, ai_sport, etc.)
        that publish commands on the same low-level topics we want to control.
        Leaving them running causes conflicting commands. Blocks until CheckMode
        reports no active mode. In simulation, no motion service is running, so
        CheckMode returns None and we exit immediately.
        """
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self._released_mode_name = None  # remembered so stop() can restore it
        while True:
            status, result = self.msc.CheckMode()
            if result is None:
                # No motion service is running (typical in simulation). Nothing to release.
                return
            if not result.get("name"):
                return
            self._released_mode_name = result["name"]
            self.msc.ReleaseMode()
            time.sleep(1)

    def _create_publishers(self):
        self.low_cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_publisher.Init()
        if self.has_hands:
            self.left_hand_cmd_publisher = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
            self.left_hand_cmd_publisher.Init()
            self.right_hand_cmd_publisher = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
            self.right_hand_cmd_publisher.Init()

    def _create_subscribers(self):
        # queueLen=1: we only ever care about the latest state.
        self.low_state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_subscriber.Init(self._on_low_state_received, queueLen=1)
        if self.has_hands:
            self.left_hand_state_subscriber = ChannelSubscriber("rt/dex3/left/state", HandState_)
            self.left_hand_state_subscriber.Init(self._on_left_hand_state_received, queueLen=1)
            self.right_hand_state_subscriber = ChannelSubscriber("rt/dex3/right/state", HandState_)
            self.right_hand_state_subscriber.Init(self._on_right_hand_state_received, queueLen=1)

    def _on_low_state_received(self, msg: LowState_):
        self.low_state = msg
        if not self.updated_mode_machine:
            self.mode_machine = msg.mode_machine
            self.updated_mode_machine = True

    def _on_left_hand_state_received(self, msg: HandState_):
        self.left_hand_state = msg

    def _on_right_hand_state_received(self, msg: HandState_):
        self.right_hand_state = msg

    def start(self):
        self._wait_until_mode_machine_is_updated()
        if self.has_hands:
            self._wait_until_hand_states_received()
        self.policy.reset(self._build_state())
        self._create_threads()
        self._start_threads()

    def _wait_until_mode_machine_is_updated(self, timeout_s: float = 5.0):
        """Block until the first LowState arrives and populates self.mode_machine.

        The robot rejects commands whose mode_machine does not match its own
        published value. This acts as a hardware-variant handshake and forces
        the client to have received state before publishing commands.
        """
        start = time.monotonic()
        while not self.updated_mode_machine:
            if time.monotonic() - start > timeout_s:
                raise RuntimeError(
                    "No LowState received within timeout. "
                    "Check network interface and that the robot is powered on."
                )
            time.sleep(0.01)

    def _wait_until_hand_states_received(self, timeout_s: float = 5.0):
        """Block until both Dex3 hand state topics have delivered at least one message."""
        start = time.monotonic()
        while self.left_hand_state is None or self.right_hand_state is None:
            if time.monotonic() - start > timeout_s:
                raise RuntimeError(
                    "Dex3 hand states not received within timeout. "
                    "Check that both hands are powered and connected."
                )
            time.sleep(0.01)

    def _build_state(self) -> G1State:
        """Snapshot current body and (if present) hand states into a G1State.

        Reads are atomic on the references themselves (CPython attribute access),
        so we capture each one and let the policy see a consistent bundle. The
        individual messages may be from slightly different DDS ticks, which is
        expected and harmless for a 10-100 Hz policy.
        """
        if self.has_hands:
            return G1State(
                body=self.low_state,
                left_hand=self.left_hand_state,
                right_hand=self.right_hand_state,
            )
        return G1State(body=self.low_state)

    def _create_threads(self):
        self.control_thread = RecurrentThread(
            interval=self.control_dt, target=self._control_step, name="control"
        )
        self.policy_thread = RecurrentThread(
            interval=self.policy.dt, target=self._policy_step, name="policy"
        )

    def _start_threads(self):
        # Run the policy once synchronously so the tracker has a target on its first tick.
        self._policy_step()
        self.policy_thread.Start()
        self.control_thread.Start()

    def _policy_step(self):
        if self.low_state is None:
            return
        if self.has_hands and (self.left_hand_state is None or self.right_hand_state is None):
            return
        action = self.policy.step(self._build_state())
        with self._target_lock:
            self._target_action = action

    def _control_step(self):
        with self._target_lock:
            action = self._target_action
        if action is None:
            return
        self._apply_body_action(action)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.low_cmd_publisher.Write(self.low_cmd)
        if self.has_hands:
            self._apply_hand_actions(action)
            self.left_hand_cmd_publisher.Write(self.left_hand_cmd)
            self.right_hand_cmd_publisher.Write(self.right_hand_cmd)

    def _apply_body_action(self, action: JointAction):
        # PR (0): series pitch/roll ankle control — the only mode any deployed
        # policy uses. (AB (1) = parallel A/B actuator semantics, unused.)
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.mode_machine
        kp = action.kp if action.kp is not None else DEFAULT_KP
        kd = action.kd if action.kd is not None else DEFAULT_KD
        for i in range(BODY_NUM_MOTORS):
            mc = self.low_cmd.motor_cmd[i]
            mc.mode = 1  # enable
            mc.q = float(action.q[i])
            mc.dq = float(action.dq[i]) if action.dq is not None else 0.0
            mc.tau = float(action.tau[i]) if action.tau is not None else 0.0
            mc.kp = float(kp[i])
            mc.kd = float(kd[i])

    def _apply_hand_actions(self, action: JointAction):
        """Fill both hand command messages from action slices q[29:36] and q[36:43]."""
        for hand_cmd, hand_slice in (
            (self.left_hand_cmd, LEFT_HAND_SLICE),
            (self.right_hand_cmd, RIGHT_HAND_SLICE),
        ):
            kp = action.kp[hand_slice] if action.kp is not None else DEFAULT_HAND_KP
            kd = action.kd[hand_slice] if action.kd is not None else DEFAULT_HAND_KD
            q = action.q[hand_slice]
            dq = action.dq[hand_slice] if action.dq is not None else None
            tau = action.tau[hand_slice] if action.tau is not None else None
            for i in range(HAND_NUM_MOTORS):
                mc = hand_cmd.motor_cmd[i]
                mc.mode = self._encode_dex3_motor_mode(motor_id=i, status=1, timeout=True)
                mc.q = float(q[i])
                mc.dq = float(dq[i]) if dq is not None else 0.0
                mc.tau = float(tau[i]) if tau is not None else 0.0
                mc.kp = float(kp[i])
                mc.kd = float(kd[i])
            hand_cmd.reserve[0] = 0

    def _encode_dex3_motor_mode(self, motor_id: int, status: int = 1, timeout: bool = True) -> int:
        """Pack a Dex3 motor mode byte (RIS_Mode_t).

        status: 0 = Lock, 1 = FOC (normal control).
        timeout: True enables the 1s watchdog. Hands fault into a safe state
        if the master stops sending commands for >1s.
        """
        return (motor_id & 0x0F) | ((status & 0x07) << 4) | ((1 if timeout else 0) << 7)

    def stop(self):
        if hasattr(self, "policy_thread"):
            self.policy_thread.Wait(1.0)   # sets the quit flag and joins
            self.control_thread.Wait(1.0)
        if getattr(self, "_released_mode_name", None):
            self.msc.SelectMode(self._released_mode_name)