"""Observation assembly — must match the deployed ONNX policy input.

The policy expects 45 floats in this exact order and per-term scaling
(source: rl/olaf_bipedal_robot/tasks/manager_based/olaf_locomotion/
 olaf_env_cfg.py:120-135 ObservationsCfg.PolicyCfg):

    idx range │ term                        │ scale │ shape
    ──────────┼─────────────────────────────┼───────┼───────
       0:3    │ imu_ang_vel                 │ 0.20  │ (3,)
       3:6    │ imu_projected_gravity       │ 1.00  │ (3,)
       6:18   │ joint_pos_rel = q − q0      │ 1.00  │ (12,)
      18:30   │ joint_vel_rel × 0.05        │ 0.05  │ (12,)
      30:42   │ last_action (raw, pre-scale)│ 1.00  │ (12,)
      42:45   │ velocity_command (vx,vy,wz) │ 1.00  │ (3,)

Total: 45 floats.
"""
from __future__ import annotations

import numpy as np

from config import DEFAULT_JOINT_POS, JOINT_ORDER, N_JOINTS

OBS_DIM = 3 + 3 + N_JOINTS + N_JOINTS + N_JOINTS + 3  # 45

# Per-term scales from olaf_env_cfg.py — must match training exactly.
_ANG_VEL_SCALE = 0.20
_JOINT_VEL_SCALE = 0.05


class ObservationBuilder:
    def __init__(self) -> None:
        self.last_action = np.zeros(N_JOINTS, dtype=np.float32)
        assert len(JOINT_ORDER) == N_JOINTS

    def build(self,
              velocity_cmd: np.ndarray,        # (3,) [vx, vy, wz]
              base_ang_vel_root: np.ndarray,   # (3,) rad/s, base_link frame
              projected_gravity: np.ndarray,   # (3,) unit vec, base_link frame
              joint_pos: np.ndarray,           # (12,) URDF frame, rad
              joint_vel: np.ndarray,           # (12,) URDF frame, rad/s
              ) -> np.ndarray:
        joint_pos_rel = (joint_pos - DEFAULT_JOINT_POS).astype(np.float32)

        obs = np.concatenate([
            base_ang_vel_root.astype(np.float32) * _ANG_VEL_SCALE,
            projected_gravity.astype(np.float32),
            joint_pos_rel,
            joint_vel.astype(np.float32) * _JOINT_VEL_SCALE,
            self.last_action,
            velocity_cmd.astype(np.float32),
        ])
        assert obs.shape == (OBS_DIM,), f"obs dim {obs.shape} != {OBS_DIM}"
        return obs

    def push_action(self, action: np.ndarray) -> None:
        self.last_action = action.astype(np.float32).copy()