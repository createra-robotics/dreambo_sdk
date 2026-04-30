"""IMU verification tool — same data path as `run.py` IDLE mode, no joystick.

Reuses the production `Imu` class from `imu.py`, so the printed values are
the SAME ones the policy will consume:

    * `ang_vel`           — base-frame angular velocity (rad/s), after
                            `R_base_from_imu` is applied.
    * `proj_g`            — projected gravity in base frame, after
                            `_projected_gravity_from_quat` (-row₂(R)) and
                            `R_base_from_imu`.
    * `yaw`               — chip-reported heading (rad).

Two views per tick:

    1. The post-transform line — exactly what `run.py` IDLE prints. Use this
       for the four tilt tests (forward / left / static / yaw) to verify
       that `R_base_from_imu` is correct.
    2. A "raw chip" line that reports `accel`, `quaternion`, and the
       chip-frame `proj_g` *before* `R_base_from_imu` is applied. Useful
       for sanity-checking the chip itself.

Usage:
    python imu_dump.py                 # default port + baud, 5 Hz print
    python imu_dump.py --rate 10
    python imu_dump.py --port /dev/ttyUSB0 --bps 921600
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from imu import (
    DEFAULT_BAUD,
    DEFAULT_PORT,
    Imu,
    _projected_gravity_from_quat,
)


def _format_vec(v: np.ndarray) -> str:
    return f"x={float(v[0]):+.3f}, y={float(v[1]):+.3f}, z={float(v[2]):+.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify IMU mounting against base_link without the joystick. "
                    "Prints the same ang_vel / proj_g / yaw that run.py IDLE shows."
    )
    parser.add_argument("--port", default=DEFAULT_PORT,
                        help=f"Serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--bps", type=int, default=DEFAULT_BAUD,
                        help=f"Baud rate (default: {DEFAULT_BAUD})")
    parser.add_argument("--rate", type=float, default=5.0,
                        help="Print rate in Hz (default: 5.0)")
    parser.add_argument("--no-raw", action="store_true",
                        help="Suppress the raw-chip diagnostic line")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    print("=" * 80)
    print(f"Port      : {args.port}")
    print(f"Baud      : {args.bps}")
    print(f"Print rate: {args.rate:.1f} Hz")
    print("Reusing imu.Imu — same transform path as run.py IDLE mode.")
    print()
    print("Tilt-test reference (with mounting correctly aligned to base_link):")
    print("  static level    :  proj_g ≈ (x=+0.000, y=+0.000, z=-1.000)")
    print("  forward tilt 10°:  proj_g_x → +0.17    (nose down)")
    print("  left tilt 10°   :  proj_g_y → -0.17    (left ear toward floor)")
    print("  yaw CCW (top-view): ang_vel_z is positive")
    print("=" * 80)
    print("Press Ctrl+C to stop.\n")

    imu = Imu(port=args.port, baudrate=args.bps)

    interval = 1.0 / max(args.rate, 0.1)
    next_t = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            next_t = now + interval

            ang_vel, proj_g, yaw = imu.read()
            ts = time.strftime("%H:%M:%S", time.localtime())
            # Post-transform line — same format as run.py IDLE.
            print(
                f"[{ts}] imu: "
                f"ang_vel=[{_format_vec(ang_vel)}]  "
                f"proj_g=[{_format_vec(proj_g)}]  "
                f"yaw={yaw:+.3f}"
            )

            if args.no_raw:
                continue
            accel, ang_vel_chip, quat = imu.read_raw()
            qw, qx, qy, qz = (float(quat[0]), float(quat[1]),
                              float(quat[2]), float(quat[3]))
            proj_g_chip = _projected_gravity_from_quat(qw, qx, qy, qz)
            print(
                f"        raw: accel=[{_format_vec(accel)}]  "
                f"ang_vel_chip=[{_format_vec(ang_vel_chip)}]  "
                f"quat=[w={qw:+.3f}, x={qx:+.3f}, y={qy:+.3f}, z={qz:+.3f}]  "
                f"proj_g_chip=[{_format_vec(proj_g_chip)}]"
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        imu.close()


if __name__ == "__main__":
    sys.exit(main())