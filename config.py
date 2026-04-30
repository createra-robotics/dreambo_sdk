"""Wiring / kinematics constants for the physical OLAF.

Keep in sync with ``olaf_walking_env_cfg.py``.  The ORDER of ``JOINT_ORDER``
is the contract between policy outputs, observations, and motor commands.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np


# ---------------------------------------------------------------------------
# Joint order — MUST match Isaac Lab's `robot.data.joint_names`, which
# interleaves [L, R] per family. Verified against a `play.py` print-out from
# the training machine on 2026-04-28 — keep this list in sync if the URDF
# changes. The same order pins observation indexing AND motor commands.
# ---------------------------------------------------------------------------
JOINT_ORDER = (
    "l_hip_yaw_joint",
    "r_hip_yaw_joint",
    "l_hip_roll_joint",
    "r_hip_roll_joint",
    "l_hip_pitch_joint",
    "r_hip_pitch_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
)
N_JOINTS = len(JOINT_ORDER)

# Action scale baked into the training env's JointPositionActionCfg
# (rl/olaf_bipedal_robot/tasks/manager_based/olaf_locomotion/olaf_env_cfg.py:75,
# scale=0.5, use_default_offset=True). The policy outputs raw actions; the
# SDK must reproduce `q_target = ACTION_SCALE * action + q0` exactly, or
# every policy step is amplified 2x and the robot falls.
ACTION_SCALE = 0.5

# Default joint pose `q0` (the bent-knee stance the policy was trained
# around). q_target = q0 + ACTION_SCALE * action. L and R share numeric
# values; URDF axis flip handles the physical mirror, do NOT sign-flip
# targets here.
DEFAULT_JOINT_POS = np.array([
    0.000, 0.000,   # hip_yaw    (L, R)
    0.000, 0.000,   # hip_roll   (L, R)
    0.900, 0.900,   # hip_pitch  (L, R)
    1.650, 1.650,   # knee_pitch (L, R)
    0.700, 0.700,   # ankle_pitch(L, R)
    0.000, 0.000,   # ankle_roll (L, R)
], dtype=np.float32)

# URDF/MJCF hard joint limits (lower, upper). Symmetric L/R; source:
# rl/assets/URDF/olaf_robstride/urdf/olaf_robstride.urdf.
JOINT_LIMITS = np.array([
    (-0.2600,  0.2600),  # l_hip_yaw
    (-0.2600,  0.2600),  # r_hip_yaw
    (-0.2618,  0.3490),  # l_hip_roll
    (-0.2618,  0.3490),  # r_hip_roll
    (-0.1000,  1.4000),  # l_hip_pitch
    (-0.1000,  1.4000),  # r_hip_pitch
    (-0.4000,  2.0000),  # l_knee_pitch
    (-0.4000,  2.0000),  # r_knee_pitch
    (-0.6500,  1.0500),  # l_ankle_pitch
    (-0.6500,  1.0500),  # r_ankle_pitch
    (-0.2000,  0.2600),  # l_ankle_roll
    (-0.2000,  0.2600),  # r_ankle_roll
], dtype=np.float32)

# Training applies `soft_joint_pos_limit_factor=0.95` (rl/.../olaf.py:120)
# which shrinks each joint's range to 95% around its midpoint. The policy
# was never asked to drive past these soft limits, so clipping `q_target` to
# JOINT_LIMITS_SOFT (rather than the URDF hard stop) keeps the SDK inside
# the training distribution and leaves a 5% mechanical margin before the
# physical end-stop. Hard JOINT_LIMITS stays around as a last-resort guard.
SOFT_LIMIT_FACTOR = 0.95
_jl_mid = (JOINT_LIMITS[:, 0] + JOINT_LIMITS[:, 1]) * 0.5
_jl_half = (JOINT_LIMITS[:, 1] - JOINT_LIMITS[:, 0]) * 0.5 * SOFT_LIMIT_FACTOR
JOINT_LIMITS_SOFT = np.stack([_jl_mid - _jl_half, _jl_mid + _jl_half], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Motor hardware classes
# ---------------------------------------------------------------------------
class MotorKind(Enum):
    ROBSTRIDE_RS00 = "rs00"   # ankle pitch/roll
    ROBSTRIDE_RS02 = "rs02"   # hip yaw
    ROBSTRIDE_RS03 = "rs03"   # hip roll/pitch, knee


@dataclass(frozen=True)
class MotorSpec:
    can_id: int             # arbitration ID on the bus
    kind: MotorKind
    direction: int = 1      # +1 or -1 to flip sign vs. URDF convention
    zero_offset_rad: float = 0.0  # mechanical zero offset (encoder − URDF zero)
    kp: float = 0.0
    kd: float = 0.0
    # Software-side torque cap (N·m, |τ|). Mirrors the training-side
    # `effort_limit_sim` per actuator family:
    #   rl/olaf_bipedal_robot/robots/olaf.py:effort_limit_sim
    #     RS02 (hip_yaw)         11.9 N·m
    #     RS03 (hip_roll/pitch,  42.0 N·m
    #           knee)
    #     RS00 (ankle_pitch/     14.0 N·m
    #           ankle_roll)
    # The Robstride firmware closes its inner PD loop itself, so we cannot
    # directly clamp the firmware's torque output. `motors.command` enforces
    # this cap by shrinking `(q_des − q_meas)` such that the estimated
    # `kp*Δq + kd*Δqd` stays within tau_limit — keeps the trained gains
    # honest without going outside the policy's training distribution.
    tau_limit: float = 0.0


# ---------------------------------------------------------------------------
# Motor wiring — IDs match the layout.
#
# kp/kd per actuator family must reproduce the trained gains
# (rl/olaf_bipedal_robot/robots/olaf.py: RS02 40/3.000, RS03 78.957/5.027,
# RS00 16.581/1.056) so the policy doesn't see a different plant than it
# trained against.
#
# WARNING: the `direction` flags below were calibrated against the OLD
# asymmetric `DEFAULT_JOINT_POS` (which sign-flipped right-leg targets in
# URDF frame). The constants above now match the new training defaults
# (symmetric L/R values, mirroring lives in URDF axes). If the right leg
# drives in the wrong direction at boot ramp, the right-side `direction`
# calibrations (esp. r_hip_pitch_joint, r_knee_pitch_joint) likely need to
# be re-verified on bench via single-joint slomo runs.
# ---------------------------------------------------------------------------

_RS02_KP, _RS02_KD = 40.000, 3.000   # hip_yaw
_RS03_KP, _RS03_KD = 78.957, 5.027   # hip_roll, hip_pitch, knee_pitch
_RS00_KP, _RS00_KD = 16.581, 1.056   # ankle_pitch, ankle_roll

# Per-family torque caps (N·m) — mirror training `effort_limit_sim`.
_RS02_TAU = 11.9   # hip_yaw
_RS03_TAU = 42.0   # hip_roll, hip_pitch, knee_pitch
_RS00_TAU = 14.0   # ankle_pitch, ankle_roll

MOTOR_TABLE: dict[str, MotorSpec] = {
    "l_hip_yaw_joint":     MotorSpec(can_id=1,  kind=MotorKind.ROBSTRIDE_RS02, direction=1, kp=_RS02_KP, kd=_RS02_KD, tau_limit=_RS02_TAU),
    "l_hip_roll_joint":    MotorSpec(can_id=2,  kind=MotorKind.ROBSTRIDE_RS03, direction=1, kp=_RS03_KP, kd=_RS03_KD, tau_limit=_RS03_TAU),
    "l_hip_pitch_joint":   MotorSpec(can_id=3,  kind=MotorKind.ROBSTRIDE_RS03, direction=1, kp=_RS03_KP, kd=_RS03_KD, tau_limit=_RS03_TAU),
    "l_knee_pitch_joint":  MotorSpec(can_id=4,  kind=MotorKind.ROBSTRIDE_RS03, direction=-1, kp=_RS03_KP, kd=_RS03_KD, tau_limit=_RS03_TAU),
    "l_ankle_pitch_joint": MotorSpec(can_id=5,  kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=_RS00_KP, kd=_RS00_KD, tau_limit=_RS00_TAU),
    "l_ankle_roll_joint":  MotorSpec(can_id=6,  kind=MotorKind.ROBSTRIDE_RS00, direction=1, kp=_RS00_KP, kd=_RS00_KD, tau_limit=_RS00_TAU),

    "r_hip_yaw_joint":     MotorSpec(can_id=7,  kind=MotorKind.ROBSTRIDE_RS02, direction=1, kp=_RS02_KP, kd=_RS02_KD, tau_limit=_RS02_TAU),
    "r_hip_roll_joint":    MotorSpec(can_id=8,  kind=MotorKind.ROBSTRIDE_RS03, direction=1,  kp=_RS03_KP, kd=_RS03_KD, tau_limit=_RS03_TAU),
    "r_hip_pitch_joint":   MotorSpec(can_id=9,  kind=MotorKind.ROBSTRIDE_RS03, direction=1,  kp=_RS03_KP, kd=_RS03_KD, tau_limit=_RS03_TAU),
    "r_knee_pitch_joint":  MotorSpec(can_id=10, kind=MotorKind.ROBSTRIDE_RS03, direction=-1, kp=_RS03_KP, kd=_RS03_KD, tau_limit=_RS03_TAU),
    "r_ankle_pitch_joint": MotorSpec(can_id=11, kind=MotorKind.ROBSTRIDE_RS00, direction=-1, kp=_RS00_KP, kd=_RS00_KD, tau_limit=_RS00_TAU),
    "r_ankle_roll_joint":  MotorSpec(can_id=12, kind=MotorKind.ROBSTRIDE_RS00, direction=1, kp=_RS00_KP, kd=_RS00_KD, tau_limit=_RS00_TAU),
}

# ---------------------------------------------------------------------------
# Loop rates
# ---------------------------------------------------------------------------
POLICY_HZ       = 50.0
# 200 Hz gives 5× Nyquist headroom over the 37.5 Hz LPF cutoff and keeps
# total CAN traffic (12 motors × (cmd + reply)) under the 1 Mbps bus.
# 600 Hz saturated the bus once the Robstride library started receiving
# real status replies from the motors.
MOTOR_HZ        = 200.0
POLICY_DT       = 1.0 / POLICY_HZ
MOTOR_DT        = 1.0 / MOTOR_HZ
LPF_CUTOFF_HZ   = 37.5                      # per paper §VII-C

# Safety
POLICY_WATCHDOG_S = 0.040                   # drop to damping if policy misses
KP_RAMP_S         = 2.0                     # soft-start ramp on startup

# Max per-joint speed (rad/s) when ramping to zero pose (X) or
# DEFAULT_JOINT_POS (Y). The ramp is *synchronized*: every joint arrives at
# its goal simultaneously — the joint with the longest travel moves at this
# speed, shorter-travel joints move slower in proportion.
POSE_SPEED_RAD_S  = 0.8

# Slow-motion debug: per-joint velocity cap applied to the policy target
# before it's published to the motor thread. Enabled via `run.py --slomo`.
SLOMO_VMAX_RAD_S  = 0.5

# ---------------------------------------------------------------------------
# CAN-FD
# ---------------------------------------------------------------------------
# When True, all outgoing frames (Robstride + Damiao) are promoted to CAN-FD
# with bitrate-switch enabled. Requires THREE things configured elsewhere:
#   1. Kernel interface in FD mode:
#        sudo ip link set can_usb down
#        sudo ip link set can_usb type can bitrate 1000000 dbitrate 5000000 fd on
#        sudo ip link set can_usb up txqueuelen 1000
#   2. Each motor's firmware configured for CAN-FD at the chosen data bitrate
#      (Robstride + Damiao both support it, but must be enabled per-motor).
#   3. A CAN adapter that supports FD (CANable at 5 Mbps is fine).
# Leave False until all three are true — otherwise every send raises.
CAN_FD_ENABLED    = False
CAN_FD_DBITRATE   = 5_000_000
