"""Thin SocketCAN wrapper.

Drop-in replacement for vendor USB-serial protocols is out of scope — if
your adapter (e.g. Robstride's stock dongle) needs a framed UART protocol,
subclass :class:`CanBus` and override ``send`` / ``recv``.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import can  # python-can

log = logging.getLogger(__name__)


@dataclass
class CanFrame:
    arbitration_id: int
    data: bytes
    is_extended_id: bool = False


class CanBus:
    def __init__(self, channel: str = "can_usb", bitrate: int = 1_000_000):
        self._bus = can.interface.Bus(
            channel=channel,
            interface="socketcan",
            bitrate=bitrate,
            receive_own_messages=False,
        )
        self._tx_lock = threading.Lock()

    def send(self, frame: CanFrame, timeout: float = 0.002) -> None:
        msg = can.Message(
            arbitration_id=frame.arbitration_id,
            data=frame.data,
            is_extended_id=frame.is_extended_id,
        )
        with self._tx_lock:
            self._bus.send(msg, timeout=timeout)

    def recv(self, timeout: float = 0.001) -> CanFrame | None:
        msg = self._bus.recv(timeout=timeout)
        if msg is None:
            return None
        return CanFrame(msg.arbitration_id, bytes(msg.data), msg.is_extended_id)

    def close(self) -> None:
        self._bus.shutdown()
