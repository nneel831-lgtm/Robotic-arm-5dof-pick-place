#!/usr/bin/env python3
"""
gripper_driver.py
==================
Thin wrapper around the Hiwonder/LX-16A-style serial bus servo protocol
driving the gripper (Servo ID 6). Exposes open_gripper()/close_gripper()
as plain functions so dynamic_ik_tracker.py can import and call them
directly, instead of spawning your two standalone scripts as subprocesses.

Why this matters here:
  - subprocess.run(["python3", "close_gripper.py"]) works, but it re-opens
    the serial port and re-pays the 1s "let the adapter wake up" sleep on
    EVERY call. Importing the function and opening the port only for the
    duration of the move keeps the same safety behavior (port isn't held
    open indefinitely) without the repeated wake-up tax if you ever call
    this at higher rates than "once per grasp".
  - Having exactly one close_gripper() implementation means your open/close
    positions can't silently drift apart (see the bug below).

*** BUG FIXED FROM YOUR ORIGINAL SCRIPTS ***
Your "open" and "close" scripts both moved Servo 6 to position 350. They
were identical except for the print string -- "closing" was actually just
re-sending the open command. GRIPPER_CLOSED_POS below is a placeholder;
you MUST measure your gripper's real closed position (jog the servo by
hand with a quick test script, e.g. try 550, 650, 700, watch the jaws,
and log the exact value) and set it correctly before relying on this.
"""

import serial
import time

SERIAL_PORT = '/dev/ttyUSB1'
BAUD_RATE = 9600
GRIPPER_SERVO_ID = 6

GRIPPER_OPEN_POS = 310     # confirmed from your working open script
GRIPPER_CLOSED_POS = 450   # PLACEHOLDER -- measure and replace this value


def _build_move_packet(servo_id: int, position: int, move_time_ms: int = 500) -> bytearray:
    return bytearray([
        0x55, 0x55,
        8,           # data length (5 + 3*1)
        0x03,        # command: MOVE
        1,           # number of servos in this packet
        move_time_ms & 0xFF, (move_time_ms >> 8) & 0xFF,
        servo_id, position & 0xFF, (position >> 8) & 0xFF,
    ])


def _send_move(position: int, move_time_ms: int = 500, port: str = SERIAL_PORT) -> bool:
    try:
        with serial.Serial(port, BAUD_RATE, timeout=1) as ser:
            time.sleep(1.0)  # let the adapter wake up -- same as your original scripts
            ser.write(_build_move_packet(GRIPPER_SERVO_ID, position, move_time_ms))
            time.sleep(move_time_ms / 1000.0 + 0.1)  # block until the physical move finishes
        return True
    except serial.SerialException as e:
        print(f"[gripper_driver] Serial port error: {e}")
        return False


def open_gripper(move_time_ms: int = 500) -> bool:
    print(f"[gripper_driver] Opening -> pos {GRIPPER_OPEN_POS}")
    return _send_move(GRIPPER_OPEN_POS, move_time_ms)


def close_gripper(move_time_ms: int = 500) -> bool:
    print(f"[gripper_driver] Closing -> pos {GRIPPER_CLOSED_POS}")
    return _send_move(GRIPPER_CLOSED_POS, move_time_ms)


if __name__ == "__main__":
    # Manual bench test -- run this standalone first to confirm
    # GRIPPER_CLOSED_POS actually closes the jaws before trusting it
    # inside the ROS2 node.
    print("Opening...")
    open_gripper()
    time.sleep(1.5)
    print("Closing...")
    close_gripper()
