"""Main runtime entry.

Startup procedure (Xbox-compatible controller required):

    1. Suspend the robot or hold it so the legs hang freely.
    2. ``python run.py`` — motors enable, legs stay compliant (kp = 0).
    3. Press **Y** to ramp every joint to ``DEFAULT_JOINT_POS`` (the
       training-env stance the policy expects).
    4. Once settled at the default pose, press **B** to hand control over
       to the ONNX policy. The left stick drives ``velocity_cmd``.

Buttons:

    * **X** — ramp every joint back to 0 rad (URDF zero pose).
    * **Y** — ramp every joint to ``DEFAULT_JOINT_POS`` (squat pose).
    * **B** — hand control to the ONNX policy (gated by --pose-tolerance).
    * **A** — EMERGENCY STOP: latch, damp motors, exit.
    * **RB** — encoder-zero calibration. From IDLE, press RB once to enter
      CALIBRATE: motors go limp, the console streams every joint's q at
      5 Hz, and you rotate each joint by hand to the pose you want to
      call "0 rad". Press RB again to write the zero into Robstride flash
      (firmware command 0x06 + SAVE 0x16 → survives power cycle, no
      config file needed). An audit record is written to
      ``calibration.json`` for diagnostics only.

Command-line arguments:

    --can-usb CHANNEL        (default: ``can_usb``)
    --can-spi CHANNEL        (default: ``can_spi``)
        CAN interface names. Left leg (motor IDs 1–6) goes on --can-usb,
        right leg (IDs 7–12) goes on --can-spi.

    --bus {both,usb,spi}     (default: ``both``)
        Restrict motor commands to a single CAN bus. The ONNX policy
        still runs at full dimension; joints on the inactive bus stay
        pinned at ``DEFAULT_JOINT_POS`` in the observation.

    --log-level LEVEL        (default: ``INFO``)
        Python logging level (DEBUG / INFO / WARNING / ERROR).

    --pose-tolerance RAD     (default: 0.15)
        Max L∞ joint error vs ``DEFAULT_JOINT_POS`` allowed for B to
        start the policy. Press Y first and wait until it settles.

    --slomo
        Clamp per-joint target slew during POLICY mode to
        ``SLOMO_VMAX_RAD_S`` (see ``config.py``). Useful for watching a
        misbehaving policy at a safe speed.

    --debug
        After pressing B, run only --debug-actions policy steps, each
        with the per-joint target delta clamped to ±--debug-dq rad, then
        terminate cleanly. Use this to verify per-joint rotation axes
        without risking runaway motion.

    --debug-actions N        (default: 10)
    --debug-dq RAD           (default: 0.1)
        Action budget and per-action slew clamp for --debug. Only take
        effect when --debug is set.

    --joints LIST
        Comma-separated joint names (with or without the ``_joint``
        suffix) or indices (0–11) to drive from the policy. Others are
        held at ``DEFAULT_JOINT_POS``. Default: all 12.
        Example: ``--joints l_knee_pitch,r_knee_pitch``.

Examples:

    python run.py                                 # normal operation
    python run.py --debug                         # 10-action axis check
    python run.py --bus usb                       # left leg only
    python run.py --slomo                          # cautious first run
    python run.py --joints l_hip_pitch,r_hip_pitch  # single-joint test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

# Suppress ONNX Runtime GPU discovery warning on headless / GPU-less boards
os.environ.setdefault("ORT_DISABLE_GPU_DEVICE_ENUMERATION", "1")

import numpy as np
from config import (
    ACTION_SCALE,
    DEFAULT_JOINT_POS,
    JOINT_LIMITS_SOFT,
    JOINT_ORDER,
    KP_RAMP_S,
    LPF_CUTOFF_HZ,
    MOTOR_DT,
    N_JOINTS,
    POLICY_DT,
    POLICY_WATCHDOG_S,
    POSE_SPEED_RAD_S,
    SLOMO_VMAX_RAD_S,
)
from imu import Imu
from joystick import BUTTON_A, BUTTON_B, BUTTON_RB, BUTTON_X, BUTTON_Y, Joystick
from motors import MotorBus
from observation import ObservationBuilder, OBS_DIM
from policy import DeployedPolicy

log = logging.getLogger("olaf.run")

ZERO_JOINT_POS = np.zeros(N_JOINTS, dtype=np.float32)


class Mode(Enum):
    IDLE = "IDLE"                 # legs hang, kp=0 (motors enabled, no stiffness)
    MOVE_ZERO = "MOVE_ZERO"       # ramp to all-zeros pose
    MOVE_DEFAULT = "MOVE_DEFAULT" # ramp to DEFAULT_JOINT_POS
    POLICY = "POLICY"             # ONNX control
    CALIBRATE = "CALIBRATE"       # motors limp, user moves joints by hand


class FirstOrderLowPass:
    """Per-joint LPF.  α = dt / (RC + dt), RC = 1 / (2π f_c)."""

    def __init__(self, cutoff_hz: float, dt: float, n: int):
        rc = 1.0 / (2.0 * np.pi * cutoff_hz)
        self.alpha = dt / (rc + dt)
        self.y = np.zeros(n, dtype=np.float32)

    def reset(self, x: np.ndarray) -> None:
        self.y = x.astype(np.float32).copy()

    def step(self, x: np.ndarray) -> np.ndarray:
        self.y += self.alpha * (x - self.y)
        return self.y


class Runtime:
    def __init__(self,
                 can_usb: str = "can_usb", can_spi: str = "can_spi",
                 bus: str = "both",
                 slomo: bool = False,
                 active_mask: np.ndarray | None = None,
                 joystick_max_vel: float = 0.6,
                 pose_tolerance: float = 0.15,
                 debug: bool = False,
                 debug_actions: int = 10,
                 debug_max_dq: float = 0.1):
        # Per-joint ramp rate when moving to zero / default pose via X or Y
        # — sourced from config.POSE_SPEED_RAD_S, no longer a CLI arg.
        self._pose_speed = float(POSE_SPEED_RAD_S)
        # B (→POLICY) is gated: current measured pos must be within this
        # L∞ tolerance of DEFAULT_JOINT_POS, else the press is rejected.
        self._pose_tolerance = float(pose_tolerance)

        # State machine — started in IDLE so the user controls when the
        # motors start applying torque.
        self._mode = Mode.IDLE
        self._mode_entered_t = time.monotonic()
        self._pose_goal = DEFAULT_JOINT_POS.copy()

        # Emergency stop — when True the motor thread keeps damping (kp=0,
        # velocity=0) every tick instead of tracking target_curr. The main
        # loop is torn down via self._stop; finally-block then disables
        # motors. Latched — there is no un-ESTOP without restarting.
        self._estop_engaged = False

        # Debug mode — sanity check for per-joint rotation direction. When
        # True, pressing B arms a bounded policy session: at most
        # ``debug_actions`` ticks of ONNX control are executed, each with
        # the per-joint target delta clamped to ±``debug_max_dq`` rad, then
        # the program terminates cleanly.
        self._debug = bool(debug)
        self._debug_max_actions = int(debug_actions)
        self._debug_max_dq = float(debug_max_dq)
        self._debug_actions_left = 0

        if bus == "both":
            active_channels = None
        elif bus == "usb":
            active_channels = (can_usb,)
        elif bus == "spi":
            active_channels = (can_spi,)
        else:
            raise ValueError(f"--bus must be one of usb/spi/both, got {bus!r}")

        # Joystick — REQUIRED. X/Y/B/A are the only control surface and A
        # is the emergency stop, so we refuse to run without a controller.
        # Opened FIRST so that a missing controller fails fast before any
        # CAN/IMU hardware is brought up.
        try:
            self._joystick = Joystick(max_lin_vel=joystick_max_vel)
            self._joystick.start()
        except Exception as e:
            raise RuntimeError(
                f"No joystick detected ({e}). Plug in an Xbox-compatible "
                f"controller and retry — operation requires A (ESTOP), "
                f"X (zero pose), Y (default pose), B (policy)."
            ) from e
        log.info("joystick attached — left stick drives velocity_cmd "
                 "(max %.2f m/s)", joystick_max_vel)

        self._motors = MotorBus(can_usb=can_usb, can_spi=can_spi,
                                active_channels=active_channels)
        self._imu = Imu()
        self._policy = DeployedPolicy()
        self._obs_builder = ObservationBuilder()

        # Shared state between policy tick and motor tick
        self._lock = threading.Lock()
        self._target_prev = DEFAULT_JOINT_POS.copy()
        self._target_curr = DEFAULT_JOINT_POS.copy()
        self._target_t = time.monotonic()
        self._kp_scale = 0.0
        self._stop = threading.Event()

        # CALIBRATE mode — set while we're talking directly to the Robstride
        # libs (SET_ZERO_POSITION + SAVE_PARAMETERS). Motor thread idles so
        # it doesn't interleave operation frames with the zeroing sequence.
        self._motor_paused = threading.Event()
        self._calib_print_t = 0.0
        # IDLE-mode q+IMU printout throttle (5 Hz).
        self._idle_print_t = 0.0

        self._lpf = FirstOrderLowPass(LPF_CUTOFF_HZ, MOTOR_DT, N_JOINTS)
        self._lpf.reset(DEFAULT_JOINT_POS)

        # Slow-motion debug: clamp |Δq_target| per policy tick.
        self._slomo_dq_max = SLOMO_VMAX_RAD_S * POLICY_DT if slomo else None
        if slomo:
            log.warning("SLOMO ENABLED — target slew capped at %.2f rad/s",
                        SLOMO_VMAX_RAD_S)

        # Active-joint mask: True where the policy's command is used.
        # Masked-out joints hold at DEFAULT_JOINT_POS with normal kp so the
        # robot stays propped up while a subset is exercised.
        if active_mask is None:
            active_mask = np.ones(N_JOINTS, dtype=bool)
        self._active_mask = active_mask.astype(bool)
        if not self._active_mask.all():
            names = [n for n, a in zip(JOINT_ORDER, self._active_mask) if a]
            log.warning("ACTIVE JOINTS: %s (others held at default)",
                        ", ".join(names) or "<none>")

    # -- motor thread (600 Hz) ---------------------------------------------
    def _motor_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            if self._motor_paused.is_set():
                time.sleep(0.005)
                next_t = time.monotonic()
                continue
            now = time.monotonic()
            with self._lock:
                target_prev = self._target_prev
                target_curr = self._target_curr
                t0 = self._target_t
                kp_scale = self._kp_scale
                stale_for = now - t0

            # ESTOP latched — skip normal command path, keep damping until
            # the main thread joins us in the finally block.
            if self._estop_engaged:
                self._motors.damp_all()
            # Watchdog — drop to damping if policy is missing ticks
            elif stale_for > POLICY_WATCHDOG_S * 2:
                self._motors.damp_all()
            else:
                # First-order hold from prev → curr over POLICY_DT
                alpha = min(max(stale_for / POLICY_DT, 0.0), 1.0)
                q_interp = (1.0 - alpha) * target_prev + alpha * target_curr
                # Safety: a NaN/inf target would blow up the MIT packer and
                # poison the LPF. Damp and resync the filter from the
                # measured joint positions so recovery is smooth.
                if not np.all(np.isfinite(q_interp)):
                    log.warning("non-finite target, damping this tick")
                    self._lpf.reset(self._motors.joint_pos)
                    self._motors.damp_all()
                else:
                    q_filt = self._lpf.step(q_interp)
                    self._motors.command(q_filt, np.zeros(N_JOINTS, dtype=np.float32),
                                         kp_scale=kp_scale)

            # Non-blocking drain of feedback
            self._motors.pump_feedback(budget_s=0.0005)

            next_t += MOTOR_DT
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # we're behind; resync

    # -- policy tick (50 Hz) -----------------------------------------------
    def _policy_tick(self) -> None:
        ang_vel, proj_g, _yaw_w = self._imu.read()
        q  = self._motors.joint_pos
        qd = self._motors.joint_vel

        obs = self._obs_builder.build(
            velocity_cmd=self._joystick.velocity_cmd,
            base_ang_vel_root=ang_vel,
            projected_gravity=proj_g,
            joint_pos=q,
            joint_vel=qd,
        )
        assert obs.shape == (OBS_DIM,)

        action = self._policy(obs)                    # raw action
        q_target = (ACTION_SCALE * action + DEFAULT_JOINT_POS).astype(np.float32)

        # Clamp to the SOFT joint limits (95% of URDF range, matching the
        # training-side `soft_joint_pos_limit_factor=0.95`). Keeps q_target
        # inside the policy's training distribution and leaves a 5%
        # mechanical margin before the physical end-stop. Applied before
        # the active mask / slomo clamp so the slew limit sees the real
        # target.
        q_target = np.clip(q_target, JOINT_LIMITS_SOFT[:, 0], JOINT_LIMITS_SOFT[:, 1])

        # Hold masked-out joints at their default; only active joints track
        # the policy. Applied before slomo clamp so the clamp sees the
        # actual per-joint delta the motors will execute.
        q_target = np.where(self._active_mask, q_target, DEFAULT_JOINT_POS)

        self._obs_builder.push_action(action)

        debug_terminate = False
        with self._lock:
            if self._slomo_dq_max is not None:
                dq = np.clip(q_target - self._target_curr,
                             -self._slomo_dq_max, self._slomo_dq_max)
                q_target = self._target_curr + dq
            if self._debug and self._debug_actions_left > 0:
                dq = np.clip(q_target - self._target_curr,
                             -self._debug_max_dq, self._debug_max_dq)
                q_target = self._target_curr + dq
                self._debug_actions_left -= 1
                done = self._debug_max_actions - self._debug_actions_left
                max_dq = float(np.max(np.abs(dq)))
                log.info("DEBUG action %d/%d — max |Δq|=%.3f rad",
                         done, self._debug_max_actions, max_dq)
                if self._debug_actions_left == 0:
                    debug_terminate = True
            self._target_prev = self._target_curr
            self._target_curr = q_target
            self._target_t = time.monotonic()
        if debug_terminate:
            log.warning("DEBUG actions exhausted — terminating")
            self._stop.set()

    # -- mode transitions ---------------------------------------------------
    def _set_mode(self, new_mode: Mode) -> None:
        if new_mode is self._mode:
            return
        self._mode = new_mode
        self._mode_entered_t = time.monotonic()
        log.info("mode → %s", new_mode.value)

    def _trigger_estop(self, reason: str) -> None:
        """Latch the ESTOP flag, best-effort damp motors, signal shutdown.

        Safe to call from any thread. Motor thread will see
        ``_estop_engaged`` on its next tick and keep damping until the
        ``_stop`` event tears down the loops and the finally-block
        disables motors.
        """
        self._estop_engaged = True
        log.error("!! EMERGENCY STOP !! %s", reason)
        try:
            self._motors.damp_all()
        except Exception as e:  # estop path — keep going no matter what
            log.error("damp_all during ESTOP raised: %s", e)
        self._stop.set()

    def _finalize_calibration(self) -> None:
        """Pause the motor thread, send SET_ZERO_POSITION + SAVE_PARAMETERS
        to every live motor, write an audit JSON, and drop back to IDLE.
        """
        log.warning("Capturing zero — pausing motor thread and writing flash.")
        self._motor_paused.set()
        # Brief settle so the in-flight motor tick can finish before we start
        # poking the same Robstride libs from the main thread.
        time.sleep(0.02)
        try:
            record = self._motors.set_zero_all()
        except Exception as e:
            log.error("set_zero_all failed: %s — dropping back to IDLE", e)
            record = {}
        finally:
            # Reseed the target state from the (now-zeroed) measured pose so
            # that resuming the motor thread doesn't snap to a stale target.
            self._motors.pump_feedback(budget_s=0.05)
            initial = self._motors.joint_pos.astype(np.float32).copy()
            with self._lock:
                self._target_prev = initial
                self._target_curr = initial.copy()
                self._target_t = time.monotonic()
                self._kp_scale = 0.0
            self._lpf.reset(initial)
            self._motor_paused.clear()

        if record:
            self._save_calibration_record(record)
        self._set_mode(Mode.IDLE)
        log.warning("Calibration complete — zero stored to motor flash. "
                    "Robot can be power-cycled without recalibration.")

    def _save_calibration_record(self, record: dict) -> None:
        """Write a human-readable audit trail of the zeroing event. The
        Robstride firmware is the source of truth for the zero — this file
        is *not* loaded back by the SDK; it exists for diagnostics and so
        you can spot-check what was captured.
        """
        path = Path(__file__).resolve().parent / "calibration.json"
        payload = {
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "note": ("Audit trail only. Zero is stored in Robstride flash "
                     "via SET_ZERO_POSITION (0x06) + SAVE_PARAMETERS (0x16) "
                     "and is NOT re-applied from this file."),
            "motors": record,
        }
        try:
            path.write_text(json.dumps(payload, indent=2) + "\n")
            log.info("calibration audit written to %s", path)
        except Exception as e:
            log.warning("failed to write %s: %s", path, e)

    def _on_button(self, btn: int) -> None:
        if self._estop_engaged:
            return
        if btn == BUTTON_A:
            self._trigger_estop("A button pressed")
            return
        if btn == BUTTON_RB:
            if self._mode is Mode.CALIBRATE:
                self._finalize_calibration()
            elif self._mode is Mode.IDLE:
                log.warning("CALIBRATION MODE — motors are LIMP. Move/rotate "
                            "each joint to the desired ZERO pose, then press "
                            "RB again to write zero into Robstride flash.")
                self._calib_print_t = 0.0
                self._set_mode(Mode.CALIBRATE)
            else:
                log.warning("RB ignored — calibration only allowed from IDLE "
                            "(current mode: %s)", self._mode.value)
            return
        if btn == BUTTON_X:
            self._pose_goal = ZERO_JOINT_POS.copy()
            self._set_mode(Mode.MOVE_ZERO)
        elif btn == BUTTON_Y:
            self._pose_goal = DEFAULT_JOINT_POS.copy()
            self._set_mode(Mode.MOVE_DEFAULT)
        elif btn == BUTTON_B:
            err = float(np.max(np.abs(self._motors.joint_pos - DEFAULT_JOINT_POS)))
            if err > self._pose_tolerance:
                log.warning("B ignored — max joint error %.3f rad > tol %.3f; "
                            "press Y first to reach DEFAULT_JOINT_POS",
                            err, self._pose_tolerance)
                return
            # Sync LPF + observation state so the first policy tick starts
            # clean from the current pose.
            self._obs_builder.last_action = np.zeros(N_JOINTS, dtype=np.float32)
            self._lpf.reset(self._motors.joint_pos)
            if self._debug:
                self._debug_actions_left = self._debug_max_actions
                log.warning("DEBUG mode armed — %d policy actions @ "
                            "≤%.2f rad/joint each, then terminate.",
                            self._debug_max_actions, self._debug_max_dq)
            self._set_mode(Mode.POLICY)

    # -- per-tick dispatcher ------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()
        ramp = min((now - self._mode_entered_t) / KP_RAMP_S, 1.0)

        if self._mode is Mode.IDLE:
            # Legs hang freely. Keep target glued to measured pos so a
            # later mode transition doesn't start from a stale target.
            q_target = self._motors.joint_pos.astype(np.float32).copy()
            kp_scale = 0.0
            if now - self._idle_print_t >= 0.2:
                self._idle_print_t = now
                q = self._motors.joint_pos
                ang_vel, proj_g, yaw = self._imu.read()
                q_line = " | ".join(
                    f"{JOINT_ORDER[i].replace('_joint', '').upper()} "
                    f"[{float(q[i]):+.3f}]"
                    for i in range(N_JOINTS))
                log.info("q: %s", q_line)
                log.info(
                    "imu: ang_vel=[x=%+.3f, y=%+.3f, z=%+.3f] "
                    "proj_g=[x=%+.3f, y=%+.3f, z=%+.3f] yaw=%+.3f",
                    float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2]),
                    float(proj_g[0]),  float(proj_g[1]),  float(proj_g[2]),
                    float(yaw))
        elif self._mode is Mode.CALIBRATE:
            # Limp motors (kp=0). User rotates each joint by hand to the pose
            # that should become "0 rad". Print joint q at 5 Hz so the user
            # can see all 12 values converge toward the desired zero pose.
            q_target = self._motors.joint_pos.astype(np.float32).copy()
            kp_scale = 0.0
            if now - self._calib_print_t >= 0.2:
                self._calib_print_t = now
                q = self._motors.joint_pos
                log.info("CALIB q: %s", "  ".join(
                    f"{JOINT_ORDER[i].replace('_joint', '')}={float(q[i]):+.4f}"
                    for i in range(N_JOINTS)))
        elif self._mode in (Mode.MOVE_ZERO, Mode.MOVE_DEFAULT):
            with self._lock:
                current = self._target_curr.copy()
            # Synchronized ramp: every joint moves at a rate proportional
            # to its remaining travel so they all arrive at pose_goal at
            # the same moment. The joint with the longest travel moves at
            # pose_speed; shorter-travel joints move proportionally slower.
            # Result: hip_pitch, knee, ankle, etc. curl in concert — a
            # natural human-like squat rather than one joint racing ahead
            # while the others are still at zero.
            remaining = self._pose_goal - current
            max_abs = float(np.max(np.abs(remaining)))
            step_max = self._pose_speed * POLICY_DT
            if max_abs <= step_max:
                q_target = self._pose_goal.astype(np.float32).copy()
            else:
                dq = remaining * (step_max / max_abs)
                q_target = (current + dq).astype(np.float32)
            kp_scale = ramp
        elif self._mode is Mode.POLICY:
            # _policy_tick updates target_curr under lock itself.
            self._policy_tick()
            with self._lock:
                self._kp_scale = ramp
            return
        else:  # unreachable
            return

        with self._lock:
            self._target_prev = self._target_curr
            self._target_curr = q_target
            self._target_t = now
            self._kp_scale = kp_scale

    # -- lifecycle ----------------------------------------------------------
    def run(self) -> None:
        self._motors.enable_all()
        # Let one feedback cycle come back so target_curr seeds from real
        # measured angles (legs hanging) instead of DEFAULT_JOINT_POS.
        time.sleep(0.1)
        self._motors.pump_feedback(budget_s=0.05)
        initial = self._motors.joint_pos.astype(np.float32).copy()
        with self._lock:
            self._target_prev = initial
            self._target_curr = initial.copy()
            self._target_t = time.monotonic()
            self._kp_scale = 0.0
        self._lpf.reset(initial)

        mot = threading.Thread(target=self._motor_loop, daemon=True)
        mot.start()

        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        log.info("Ready. Buttons: X → zero pose, Y → DEFAULT_JOINT_POS, "
                 "B → ONNX policy, RB → encoder-zero calibration, "
                 "A → EMERGENCY STOP.")

        next_t = time.monotonic()
        hb_next = time.monotonic() + 1.0
        try:
            while not self._stop.is_set():
                for btn in self._joystick.consume_button_events():
                    self._on_button(btn)

                t0 = time.monotonic()
                self._tick()
                tick_s = time.monotonic() - t0
                if self._mode is Mode.POLICY and tick_s > POLICY_WATCHDOG_S:
                    log.warning("policy tick %.1f ms > watchdog", tick_s * 1e3)

                now = time.monotonic()
                if now >= hb_next:
                    hb_next = now + 1.0
                    # IDLE prints its own q+imu lines from `_tick`. Other
                    # modes still get the heartbeat so the operator can see
                    # mode / kp / vcmd / per-joint target & measurement.
                    if self._mode is not Mode.IDLE:
                        with self._lock:
                            tgt = self._target_curr.copy()
                            kp = self._kp_scale
                        vcmd = self._joystick.velocity_cmd
                        active = [(JOINT_ORDER[i], float(tgt[i]),
                                   float(self._motors.joint_pos[i]))
                                  for i, a in enumerate(self._active_mask) if a]
                        log.info(
                            "hb mode=%s kp=%.2f vx=%+.2f vy=%+.2f wz=%+.2f %s",
                            self._mode.value, kp, vcmd[0], vcmd[1], vcmd[2],
                            " ".join(f"{n}:tgt={t:+.3f},q={q:+.3f}"
                                     for n, t, q in active))

                next_t += POLICY_DT
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()
        finally:
            self._stop.set()
            mot.join(timeout=0.5)
            self._motors.disable_all()
            self._imu.close()
            self._joystick.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can-usb", default="can_usb",
                        help="CAN channel carrying motor IDs 1-6 (left leg)")
    parser.add_argument("--can-spi", default="can_spi",
                        help="CAN channel carrying motor IDs 7-12 (right leg)")
    parser.add_argument("--bus", choices=("both", "usb", "spi"), default="both",
                        help="Restrict motor commands to a single CAN bus. "
                             "The ONNX policy still runs; inactive joints "
                             "stay at DEFAULT_JOINT_POS in the observation.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--slomo", action="store_true",
                        help=f"Clamp per-joint target slew to "
                             f"SLOMO_VMAX_RAD_S ({SLOMO_VMAX_RAD_S} rad/s)")
    parser.add_argument("--pose-tolerance", type=float, default=0.15,
                        help="Max L∞ joint error (rad) vs DEFAULT_JOINT_POS "
                             "allowed for the B button to start the policy")
    parser.add_argument("--debug", action="store_true",
                        help="After pressing B, run only --debug-actions "
                             "policy steps with each per-joint target delta "
                             "clamped to ±--debug-dq rad, then terminate. "
                             "Use this to verify per-joint rotation axes "
                             "without risking runaway motion.")
    parser.add_argument("--debug-actions", type=int, default=10,
                        help="Number of policy actions to execute in --debug "
                             "mode (default: 10)")
    parser.add_argument("--debug-dq", type=float, default=0.1, help="Per-joint target delta clamp (rad) applied to "
                             "each policy action in --debug mode (default: 0.1)")
    parser.add_argument("--joints", help="Comma-separated joint names (with or without "
                             "'_joint' suffix) or indices to drive from the "
                             "policy. Others hold at DEFAULT_JOINT_POS. "
                             "Default: all 12.")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    assert len(JOINT_ORDER) == N_JOINTS
    active_mask = _parse_active_mask(args.joints)
    rt = Runtime(can_usb=args.can_usb, can_spi=args.can_spi,
                 bus=args.bus,
                 slomo=args.slomo, active_mask=active_mask,
                 pose_tolerance=args.pose_tolerance,
                 debug=args.debug,
                 debug_actions=args.debug_actions,
                 debug_max_dq=args.debug_dq)
    rt.run()


def _parse_active_mask(spec: str | None) -> np.ndarray | None:
    if spec is None:
        return None
    mask = np.zeros(N_JOINTS, dtype=bool)
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
            idx = int(tok)
            if not -N_JOINTS <= idx < N_JOINTS:
                raise SystemExit(f"--joints: index {idx} out of range")
            mask[idx] = True
            continue
        candidates = [tok, f"{tok}_joint"]
        for c in candidates:
            if c in JOINT_ORDER:
                mask[JOINT_ORDER.index(c)] = True
                break
        else:
            raise SystemExit(f"--joints: unknown joint {tok!r}. "
                             f"Known: {', '.join(JOINT_ORDER)}")
    if not mask.any():
        raise SystemExit("--joints: at least one joint required")
    return mask


if __name__ == "__main__":
    main()
