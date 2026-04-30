"""Raw dump for an Xbox One S controller — prints every button, axis, and
hat event with its index and value. Useful for discovering the mapping
before wiring a controller into `joystick.py`.

Usage:
    python joystick_dump.py                     # joystick 0, all events
    python joystick_dump.py --device 1          # pick a specific joystick
    python joystick_dump.py --axis-deadzone 0.05

Connect the controller over USB or Bluetooth (xpad / xow / bluez HID all
work). On a headless Pi, pygame is used with the dummy video driver so no
display is required.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402


# Best-effort Xbox One S name maps for the SDL2 xpad/bluez driver on Linux
# (11 buttons, 6 axes, 1 hat). Mappings vary across drivers and firmware
# versions, so these are hints — the index is the ground truth.
XBOX_BUTTON_NAMES = {
    0:  "A",
    1:  "B",
    2:  "X",
    3:  "Y",
    4:  "LB",
    5:  "RB",
    6:  "View (Back)",
    7:  "Menu (Start)",
    8:  "Xbox (Guide)",
    9:  "L3 (LeftStick click)",
    10: "R3 (RightStick click)",
}

XBOX_AXIS_NAMES = {
    0: "LeftStickX",
    1: "LeftStickY",
    2: "LT trigger",
    3: "RightStickX",
    4: "RightStickY",
    5: "RT trigger",
}


def _fmt_button(idx: int) -> str:
    name = XBOX_BUTTON_NAMES.get(idx, "?")
    return f"btn[{idx:2d}] {name}"


def _fmt_axis(idx: int) -> str:
    name = XBOX_AXIS_NAMES.get(idx, "?")
    return f"axis[{idx}] {name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0,
                        help="Joystick index (default: 0)")
    parser.add_argument("--axis-deadzone", type=float, default=0.02,
                        help="Suppress axis prints with |value| below this "
                             "(default: 0.02)")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    count = pygame.joystick.get_count()
    if count == 0:
        raise SystemExit("No joysticks detected. Connect the Xbox controller "
                         "(USB or Bluetooth) and re-run.")
    if args.device >= count:
        raise SystemExit(f"Joystick {args.device} not available "
                         f"(found {count}).")

    js = pygame.joystick.Joystick(args.device)
    js.init()

    print("=" * 72)
    print(f"Device #{args.device}: {js.get_name()}")
    print(f"  axes   : {js.get_numaxes()}")
    print(f"  buttons: {js.get_numbuttons()}")
    print(f"  hats   : {js.get_numhats()}")
    print(f"  deadzone: {args.axis_deadzone}")
    print("=" * 72)
    print("Listening for events. Ctrl+C to exit.")

    t0 = time.monotonic()
    last_axis_print: dict[int, float] = {}

    try:
        while True:
            for event in pygame.event.get():
                t = time.monotonic() - t0

                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"[{t:7.3f}] BUTTON DOWN  {_fmt_button(event.button)}")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"[{t:7.3f}] BUTTON UP    {_fmt_button(event.button)}")
                elif event.type == pygame.JOYAXISMOTION:
                    v = event.value
                    # Only print when the axis actually moved meaningfully
                    # since the last print — avoids spam from resting noise.
                    prev = last_axis_print.get(event.axis, 0.0)
                    if abs(v) >= args.axis_deadzone and abs(v - prev) >= 0.02:
                        print(f"[{t:7.3f}] AXIS         "
                              f"{_fmt_axis(event.axis)}  value={v:+.3f}")
                        last_axis_print[event.axis] = v
                    elif abs(v) < args.axis_deadzone <= abs(prev):
                        # Returned to rest — print once so you can see release
                        print(f"[{t:7.3f}] AXIS         "
                              f"{_fmt_axis(event.axis)}  value={v:+.3f} (rest)")
                        last_axis_print[event.axis] = v
                elif event.type == pygame.JOYHATMOTION:
                    hx, hy = event.value
                    print(f"[{t:7.3f}] HAT          hat[{event.hat}]  "
                          f"value=({hx:+d}, {hy:+d})")
                elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                    print(f"[{t:7.3f}] DEVICE       {pygame.event.event_name(event.type)}")
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        js.quit()
        pygame.quit()


if __name__ == "__main__":
    main()
