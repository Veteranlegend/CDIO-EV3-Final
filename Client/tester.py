#!/usr/bin/env pybricks-micropython

import time
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait
from pybricks.robotics import DriveBase

def start_motors(motor_a, motor_d):
    try:
        print("Starting motors: A anti-clockwise, D clockwise")
        motor_a.run(-1000)  # Anti-clockwise
        motor_d.run(1000)   # Clockwise

    except Exception as e:
        print("Error in start_motors:", e)

def start_motors_reverse(motor_a, motor_d):
    try:
        print("Starting motors: A anti-clockwise, D clockwise")
        motor_a.run(1000)  # Anti-clockwise
        motor_d.run(-1000)   # Clockwise

    except Exception as e:
        print("Error in start_motors:", e)
# This program requires LEGO EV3 MicroPython v2.0 or higher


# Function to stop the motors at ports A and D
def stop_motors(motor_a, motor_d):
    try:
        motor_a.stop()
        motor_d.stop()
        print("Motors stopped")
    except Exception as e:
        print("Error in stop_motors:", e)

def main():
    # Objects and setup
    ev3 = EV3Brick()
    left_motor = Motor(Port.B)
    right_motor = Motor(Port.C)
    motor_a = Motor(Port.A)
    motor_d = Motor(Port.D)

    # Correct wheel diameter and axle track (in millimeters)
    wheel_diameter = 54.5  # Make sure this is the correct measurement
    axle_track = 180  # Make sure this is the correct measurement

    # Create a DriveBase object
    robot = DriveBase(left_motor, right_motor, wheel_diameter, axle_track)

    start_motors(motor_a, motor_d)
    print("Running motors start")
    time.sleep(5)
    start_motors_reverse(motor_a, motor_d)
    time.sleep(5)
    start_motors_reverse(motor_a, motor_d)
    time.sleep(15)
    print("Runnign motors reverse")
    stop_motors(motor_a, motor_d)

if __name__ == "__main__":
    main()
