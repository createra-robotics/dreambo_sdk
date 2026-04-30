"""IMU serial driver for HI226/HI229 (or compatible) AHRS module.

Runs a dedicated reader thread at the IMU's native streaming rate
(up to 400 Hz). The thread parses every TYPE_IMU + TYPE_AHRS pair and
publishes the transformed result into a lock-protected snapshot.
``read()`` returns the latest snapshot non-blockingly, so the policy tick
(50 Hz) and any future high-rate consumer (e.g. motor thread at 600 Hz)
always see the freshest sample the hardware has produced.

The runtime expects:

    * ``ang_vel``           — 3-vector, rad/s, in the robot root frame.
    * ``projected_gravity`` — 3-vector, unit-length ``R_world→root · [0,0,-1]``.
    * ``yaw_world``         — float, current world-frame yaw (rad).

Mounting note — the IMU is rotated 180° about its Y axis relative to
``base_link`` (matching the reference C++ implementation's
``trans_axis = (-1, +1, -1)``).  The remapping is:

    robot_x = -imu_x
    robot_y =  imu_y
    robot_z = -imu_z
"""

from __future__ import annotations

import logging
import math
import struct
import threading
import numpy as np
import serial
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE


log = logging.getLogger("olaf.imu")

# IMU position relative to base_link origin, in base_link frame (metres).
# From olaf_robstride.urdf imu_joint <origin xyz="-0.095795 0 0.014"/>.
IMU_OFFSET_IN_BASE = np.array([-0.095795, 0.0, 0.014], dtype=np.float32)

# ---- Serial protocol constants (HI226/HI229 vendor protocol) ------------
FRAME_HEAD = "fc"
FRAME_END  = "fd"
TYPE_IMU   = "40"
TYPE_AHRS  = "41"
IMU_LEN    = "38"   # 56 bytes
AHRS_LEN   = "30"   # 48 bytes

# Default serial port for the CP2102 USB-UART bridgeP
DEFAULT_PORT = (
    "/dev/serial/by-id/"
    "usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
)
DEFAULT_BAUD = 921_600


def lever_arm_velocity_correction(v_imu_root: np.ndarray,
                                  ang_vel_root: np.ndarray) -> np.ndarray:
    """Transport IMU-site linear velocity to base_link origin.

    All inputs expressed in the root frame.  ``v_base = v_imu − ω × r``.
    """
    return v_imu_root - np.cross(ang_vel_root, IMU_OFFSET_IN_BASE)


def _projected_gravity_from_quat(qw: float, qx: float,
                                 qy: float, qz: float) -> np.ndarray:
    """Compute R_world→body @ (0,0,-1) from a body-to-world quaternion.

    R_world_from_body has body axes in its COLUMNS, so the body-frame
    coordinates of a world vector v are R^T @ v, which selects ROWS of R.
    For v = (0, 0, -1) we therefore need
        proj_g_body = R^T @ (0, 0, -1) = -R[2, :] = -row_2(R)
    NOT -col_2(R) (which is body_z expressed in world and has the wrong
    X/Y signs once the body is tilted). The previous version used -col_2
    and produced silently-wrong projected gravity whenever roll/pitch
    were non-zero.

    Returns the canonical "gravity straight down" if the quaternion is
    malformed (zero norm or non-finite), since a bad serial frame must
    not be allowed to poison the control loop with NaN/inf.
    """
    norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
    if not math.isfinite(norm_sq) or norm_sq < 1e-6 or norm_sq > 4.0:
        return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    row2 = np.array([
        2.0 * (qx * qz - qw * qy),       # R[2, 0]
        2.0 * (qy * qz + qw * qx),       # R[2, 1]
        1.0 - 2.0 * (qx * qx + qy * qy), # R[2, 2]
    ], dtype=np.float64)
    return (-row2).astype(np.float32)


class Imu:
    # IMU chip body frame → robot base_link frame.
    #
    # Empirical calibration (2026-04-28 dump on Olaf, robot ~level on
    # floor): chip reports accel_z ≈ -g and quaternion R[2,2] ≈ +0.996,
    # which together mean chip_z is aligned with world_z (UP). The chip
    # is therefore mounted "upright" and its body frame matches Olaf's
    # base_link REP-103 convention (forward=+x, left=+y, up=+z) directly
    # — no rotation needed.
    #
    # If a tilt-test (front-tilt vs side-tilt) reveals that proj_g_x and
    # proj_g_y are swapped, replace this with the appropriate yaw-only
    # 90°-multiple in-plane sub-matrix (e.g. [[0,-1,0],[1,0,0],[0,0,1]])
    # — but do NOT touch the z row, since the data conclusively shows
    # chip_z = +base_z.
    R_base_from_imu: np.ndarray = np.eye(3, dtype=np.float32)

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUD,
        timeout: float = 0.05,
        first_sample_timeout: float = 2.0,
    ) -> None:
        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=EIGHTBITS,
            parity=PARITY_NONE,
            stopbits=STOPBITS_ONE,
            timeout=timeout,
        )
        log.info("IMU opened on %s @ %d baud", port, baudrate)

        # Latest published sample (root frame).  Default is "stationary,
        # level" so an early read() before the first frame is harmless.
        self._ang_vel = np.zeros(3, dtype=np.float32)
        self._proj_g  = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._yaw     = 0.0
        self._sample_count = 0

        # Raw chip-frame snapshot, retained for diagnostics (e.g. imu_dump
        # verifying mounting). Updated alongside the published values under
        # the same lock so a second consumer cannot race on the serial port.
        self._raw_accel = np.zeros(3, dtype=np.float32)
        self._raw_ang_vel = np.zeros(3, dtype=np.float32)
        self._raw_quat  = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._first_sample = threading.Event()

        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="imu-reader",
        )
        self._thread.start()

        # Wait briefly for the first real sample so the policy's first
        # observation isn't based on the zero default.
        if not self._first_sample.wait(timeout=first_sample_timeout):
            log.warning("IMU: no sample within %.1f s — using defaults",
                        first_sample_timeout)

    # -- public API ---------------------------------------------------------
    def read(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (ang_vel_root[3], projected_gravity_root[3], yaw_world).

        Non-blocking snapshot of the latest sample produced by the reader
        thread. Arrays are fresh copies, safe to retain.
        """
        with self._lock:
            return self._ang_vel.copy(), self._proj_g.copy(), self._yaw

    def read_raw(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return chip-frame ``(accel[3], ang_vel[3], quat[4])`` from the
        latest IMU+AHRS pair, BEFORE ``R_base_from_imu`` is applied.

        Diagnostic-only — for tools like ``imu_dump`` that need to surface
        the raw chip output alongside ``read()``. Quat is ``[w, x, y, z]``.
        """
        with self._lock:
            return (self._raw_accel.copy(),
                    self._raw_ang_vel.copy(),
                    self._raw_quat.copy())

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._serial.is_open:
            self._serial.close()
            log.info("IMU closed (%d samples)", self._sample_count)

    # -- reader thread ------------------------------------------------------
    def _poll_loop(self) -> None:
        ser = self._serial
        got_imu = False
        got_ahrs = False
        accel_raw = np.zeros(3, dtype=np.float32)
        ang_vel_raw = np.zeros(3, dtype=np.float32)
        heading_raw = 0.0
        qw = qx = qy = qz = 0.0

        while not self._stop.is_set() and ser.is_open:
            try:
                b = ser.read(1)
            except (serial.SerialException, OSError) as e:
                log.warning("IMU serial read failed: %s", e)
                return
            if len(b) == 0:
                continue
            if b.hex() != FRAME_HEAD:
                continue

            head_type = ser.read(1).hex()
            check_len = ser.read(1).hex()

            # Skip SN, CRC8, CRC16 (4 bytes)
            ser.read(4)

            try:
                data_len = int(check_len, 16)
            except ValueError:
                continue
            data = ser.read(data_len)
            if len(data) < data_len:
                continue

            if head_type == TYPE_IMU and check_len == IMU_LEN:
                # Accelerometer is unused by the policy but cached as raw
                # diagnostic output for `imu_dump`.
                imu_data = struct.unpack("12f ii", data[:56])
                accel_raw[0] = imu_data[3]
                accel_raw[1] = imu_data[4]
                accel_raw[2] = imu_data[5]
                got_imu = True

            elif head_type == TYPE_AHRS and check_len == AHRS_LEN:
                ahrs = struct.unpack("10f ii", data[:48])
                ang_vel_raw[0] = ahrs[0]
                ang_vel_raw[1] = ahrs[1]
                ang_vel_raw[2] = ahrs[2]
                heading_raw = ahrs[5]
                qw = ahrs[6]
                qx = ahrs[7]
                qy = ahrs[8]
                qz = ahrs[9]
                got_ahrs = True

            if not (got_imu and got_ahrs):
                continue
            got_imu = False
            got_ahrs = False

            # --- Transform IMU → robot root frame ---
            if not np.all(np.isfinite(ang_vel_raw)):
                log.warning("non-finite IMU angular velocity — zeroed")
                ang_vel_raw = np.zeros(3, dtype=np.float32)

            ang_vel = self.R_base_from_imu @ ang_vel_raw
            proj_g_imu = _projected_gravity_from_quat(qw, qx, qy, qz)
            proj_g = self.R_base_from_imu @ proj_g_imu

            with self._lock:
                self._ang_vel = ang_vel.astype(np.float32)
                self._proj_g  = proj_g.astype(np.float32)
                self._yaw     = float(heading_raw)
                self._raw_accel = accel_raw.astype(np.float32).copy()
                self._raw_ang_vel = ang_vel_raw.astype(np.float32).copy()
                self._raw_quat[0] = qw
                self._raw_quat[1] = qx
                self._raw_quat[2] = qy
                self._raw_quat[3] = qz
                self._sample_count += 1
            self._first_sample.set()
