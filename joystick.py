"""Xbox One S controller → robot velocity command.

Stick mapping (ROS REP-103: +x forward, +y left, +wz CCW):
    left-stick up    →  +vx (forward)
    left-stick down  →  -vx (backward)
    left-stick left  →  +vy (left)
    left-stick right →  -vy (right)
    right-stick left →  +wz (turn left / CCW)
    right-stick right→  -wz (turn right / CW)

Stick axes are inverted from SDL's raw values (SDL: up/right = +1) to give
this convention directly.

Use from `run.py` like:

    joy = Joystick()
    joy.start()
    ...
    self._velocity_cmd = joy.velocity_cmd    # (3,) [vx, vy, wz]
    ...
    joy.stop()

Run standalone to verify mapping:

    python joystick.py
"""
from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

_AXIS_LEFT_X = 0
_AXIS_LEFT_Y = 1
# SDL right-stick X on Xbox One S over USB is index 3. Over Bluetooth on some
# kernels it's 2 — verify with `jstest /dev/input/js0` if turning feels off.
_AXIS_RIGHT_X = 3

# Training command envelope from
# rl/olaf_bipedal_robot/tasks/manager_based/olaf_locomotion/olaf_env_cfg.py:481-483.
# Anything outside is OOD for the policy, so the joystick clamps to these
# (asymmetric on vx — forward range is wider than backward).
VX_RANGE = (-0.4, 0.7)
VY_RANGE = (-0.4, 0.4)
WZ_RANGE = (-1.0, 1.0)

# SDL button indices (Xbox / Xbox One S layout)
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_LB = 4
BUTTON_RB = 5
_TRACKED_BUTTONS = (BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y, BUTTON_LB, BUTTON_RB)


class Joystick:
    """Background-polled Xbox controller. Exposes a 3-vector
    `velocity_cmd = [vx, vy, wz]` matching the policy's obs term.

    Defaults stay inside the training command envelope (PLAY: vx ∈ [-0.4, 0.7],
    vy ∈ [-0.4, 0.4], wz ∈ [-1, 1]); pushing past these is OOD.
    """

    def __init__(self,
                 device: int = 0,
                 max_lin_vel: float = 0.6,
                 max_ang_vel: float = 1.0,
                 deadzone: float = 0.08,
                 poll_hz: float = 100.0):
        self._device = device
        # Per-axis pre-clamp scale: stick in [-1, 1] is multiplied by
        # max_lin_vel / max_ang_vel, then clamped to the training envelope
        # (VX_RANGE / VY_RANGE / WZ_RANGE) so the policy never sees OOD cmds.
        self._max_v = float(max_lin_vel)
        self._max_w = float(max_ang_vel)
        self._deadzone = float(deadzone)
        self._dt = 1.0 / float(poll_hz)

        self._cmd = np.zeros(3, dtype=np.float32)   # [vx, vy, wz]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._js: pygame.joystick.JoystickType | None = None
        self._button_prev: dict[int, bool] = {b: False for b in _TRACKED_BUTTONS}
        self._button_events: list[int] = []

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= self._device:
            raise RuntimeError(
                f"Joystick {self._device} not found "
                f"(pygame sees {pygame.joystick.get_count()} devices)"
            )
        self._js = pygame.joystick.Joystick(self._device)
        self._js.init()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="joystick-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._js is not None:
            self._js.quit()
        # Tear down the joystick subsystem only — full pygame.quit() would
        # break any other pygame use in the host process.
        pygame.joystick.quit()

    # -- public state -------------------------------------------------------
    @property
    def velocity_cmd(self) -> np.ndarray:
        """(3,) float32 snapshot of [vx, vy, wz]. Safe to read."""
        with self._lock:
            return self._cmd.copy()

    def consume_button_events(self) -> list[int]:
        """Drain and return newly-pressed buttons since last call (edge-triggered)."""
        with self._lock:
            events = list(self._button_events)
            self._button_events.clear()
            return events

    # -- internals ----------------------------------------------------------
    def _apply_deadzone(self, v: float) -> float:
        if abs(v) < self._deadzone:
            return 0.0
        # Rescale outside deadzone to span [0, 1] so the stick feels full-range
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - self._deadzone) / (1.0 - self._deadzone)

    def _loop(self) -> None:
        assert self._js is not None
        while not self._stop.is_set():
            pygame.event.pump()   # required for axis/button values to refresh
            lx = self._apply_deadzone(self._js.get_axis(_AXIS_LEFT_X))
            ly = self._apply_deadzone(self._js.get_axis(_AXIS_LEFT_Y))
            rx = self._apply_deadzone(self._js.get_axis(_AXIS_RIGHT_X))
            # SDL: stick up = -1, stick right = +1. Flip to match ROS REP-103:
            # forward = +vx, left = +vy, CCW (stick left) = +wz.
            # Clamp each axis to the training command envelope so the policy
            # never sees an OOD cmd (esp. vy, where ±0.6 stick > 0.4 train).
            vx = float(np.clip(-ly * self._max_v, *VX_RANGE))
            vy = float(np.clip(-lx * self._max_v, *VY_RANGE))
            wz = float(np.clip(-rx * self._max_w, *WZ_RANGE))

            new_presses: list[int] = []
            for btn in _TRACKED_BUTTONS:
                try:
                    now_down = bool(self._js.get_button(btn))
                except Exception:
                    now_down = False
                if now_down and not self._button_prev[btn]:
                    new_presses.append(btn)
                self._button_prev[btn] = now_down

            with self._lock:
                self._cmd[0] = vx
                self._cmd[1] = vy
                self._cmd[2] = wz
                if new_presses:
                    self._button_events.extend(new_presses)
            time.sleep(self._dt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-lin-vel", type=float, default=0.6)
    parser.add_argument("--max-ang-vel", type=float, default=1.0)
    parser.add_argument("--deadzone", type=float, default=0.08)
    args = parser.parse_args()

    joy = Joystick(device=args.device,
                   max_lin_vel=args.max_lin_vel,
                   max_ang_vel=args.max_ang_vel,
                   deadzone=args.deadzone)
    joy.start()
    print(f"Joystick started. max_lin_vel={args.max_lin_vel} m/s, "
          f"max_ang_vel={args.max_ang_vel} rad/s, "
          f"deadzone={args.deadzone}. Ctrl+C to stop.")
    try:
        while True:
            cmd = joy.velocity_cmd
            print(f"\rvx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}",
                  end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        joy.stop()


if __name__ == "__main__":
    main()
