import struct


class WirelessRemoteCommander:
    DEADZONE = 0.1

    def __init__(self, policy):
        self.policy = policy

    def start(self) -> None:
        print("wireless-remote teleop: left stick = vx/vy, right stick = yaw")

    def stop(self) -> None:
        pass

    def update(self, state) -> None:
        data = bytes(state.body.wireless_remote)
        lx = struct.unpack("<f", data[4:8])[0]
        rx = struct.unpack("<f", data[8:12])[0]
        ly = struct.unpack("<f", data[20:24])[0]

        def dz(v: float) -> float:
            return v if abs(v) > self.DEADZONE else 0.0

        max_cmd = self.policy.max_cmd
        self.policy.set_command(
            dz(ly) * max_cmd[0], -dz(lx) * max_cmd[1], -dz(rx) * max_cmd[2]
        )


class KeyboardCommander:
    KEY_AXIS = {  # key -> (axis index, sign)
        "w": (0, +1.0), "s": (0, -1.0),
        "a": (1, +1.0), "d": (1, -1.0),
        "q": (2, +1.0), "e": (2, -1.0),
    }

    def __init__(self, policy):
        self.policy = policy
        self._held: set[str] = set()
        self._listener = None

    def start(self) -> None:
        if self._listener is not None:
            return
        from pynput import keyboard  # deferred: needs a display connection

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()
        print("keyboard teleop (global keys): hold w/s=vx  a/d=vy  q/e=wyaw")

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def update(self, state) -> None:
        pass  # event-driven via the pynput listener; nothing to poll per step

    @staticmethod
    def _key_name(key) -> str | None:
        char = getattr(key, "char", None)
        return char.lower() if char else None

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        if name in self.KEY_AXIS:
            self._held.add(name)
            self._apply()

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        if name in self.KEY_AXIS:
            self._held.discard(name)
            self._apply()

    def _apply(self) -> None:
        axes = [0.0, 0.0, 0.0]
        for name in self._held:
            index, sign = self.KEY_AXIS[name]
            axes[index] += sign
        max_cmd = self.policy.max_cmd
        self.policy.set_command(axes[0] * max_cmd[0], axes[1] * max_cmd[1], axes[2] * max_cmd[2])
        cmd = self.policy._cmd
        print(f"\rcmd: vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wyaw={cmd[2]:+.2f}   ",
              end="", flush=True)
