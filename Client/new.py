#!/usr/bin/env pybricks-micropython

import time
import math
import urequests
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Stop, Direction
from pybricks.robotics import DriveBase

# Correction factor for angle
correction_factor = 0.59

# Function to fetch instructions from the server
def fetch_instructions(url):
    try:
        print("Attempting to fetch instructions from the server...")
        response = urequests.get(url)
        if response.status_code == 200:
            print("Successfully fetched instructions")
            data = response.json()
            print("Received data:", data)
            return data
        else:
            print("Error fetching instructions: {}".format(response.status_code))
            return None
    except Exception as e:
        print("Exception: {}".format(e))
        return None

# Function to calculate the shortest angle between two points
def calculate_alignment_angle(blue, green, point):
    try:
        print("Calculating alignment angle with blue: {}, green: {}, point: {}".format(blue, green, point))
        angle_blue_to_point = math.degrees(math.atan2(point[1] - blue[1], point[0] - blue[0]))
        angle_green_to_point = math.degrees(math.atan2(point[1] - green[1], point[0] - green[0]))
        alignment_angle = angle_green_to_point - angle_blue_to_point

        # Normalize the angle to be within the range of [-180, 180] degrees
        while alignment_angle > 180:
            alignment_angle -= 360
        while alignment_angle < -180:
            alignment_angle += 360

        print("Calculated alignment angle: {}".format(alignment_angle))
        return alignment_angle
    except Exception as e:
        print("Error in calculate_alignment_angle:", e)
        return None

# Function to move the robot to the specified coordinates
def move_to_point(robot, distance_cm, motor_a, motor_d):
    try:
        distance_to_move = distance_cm * 10  # Convert cm to mm
        print("Moving forward {} mm".format(distance_to_move))

        # Start the motors
        start_motors(motor_a, motor_d)

        # Move the robot
        robot.straight(distance_to_move)
        robot.stop()
        time.sleep(3)
        # Stop the motors
        stop_motors(motor_a, motor_d)

        print("Finished moving forward")
    except Exception as e:
        print("Error in move_to_point:", e)

def move_to_dot_function(robot, distance_cm):
    try:
        distance_to_move = distance_cm * 10  # Convert cm to mm
        print("Moving forward {} mm".format(distance_to_move))
        # Move the robot
        robot.straight(distance_to_move)
        robot.stop()
        time.sleep(3)
        print("Finished moving forward")
    except Exception as e:
        print("Error in move_to_point:", e)

# Function to align the robot until the angles are aligned
def align_robot(robot, blue_center, green_center, point, correction_factor, server_url, alignment_tolerance=2, min_turn_angle=1, max_retries=18):
    try:
        retries = 0
        while retries < max_retries:
            alignment_angle = calculate_alignment_angle(blue_center, green_center, (point["x"], point["y"]))
            if alignment_angle is None:
                break
            print("Aligning robot with point using angle {}".format(alignment_angle))

            # Only turn the robot if the alignment angle is greater than the minimum turn angle
            if abs(alignment_angle) > min_turn_angle:
                robot.turn(alignment_angle * correction_factor)
            else:
                print("Alignment angle too small to turn: {}".format(alignment_angle))

            # Fetch updated instructions to check alignment
            instructions = fetch_instructions(server_url)
            if instructions and instructions["action"] in ["move_to_ball", "move_to_dot"]:
                blue_center = (instructions["blue_center"]["x"], instructions["blue_center"]["y"])
                green_center = (instructions["green_center"]["x"], instructions["green_center"]["y"])
                point = instructions["ball_point"] if instructions["action"] == "move_to_ball" else instructions["small_goal_dot"]

                # Recalculate angles
                new_angle_blue_to_point = math.degrees(math.atan2(point["y"] - blue_center[1], point["x"] - blue_center[0]))
                new_angle_green_to_point = math.degrees(math.atan2(point["y"] - green_center[1], point["x"] - green_center[0]))

                # Normalize angles to be within the range of [-180, 180] degrees
                while new_angle_blue_to_point > 180:
                    new_angle_blue_to_point -= 360
                while new_angle_blue_to_point < -180:
                    new_angle_blue_to_point += 360
                while new_angle_green_to_point > 180:
                    new_angle_green_to_point -= 360
                while new_angle_green_to_point < -180:
                    new_angle_green_to_point += 360

                # Check if angles are aligned
                if abs(new_angle_blue_to_point - new_angle_green_to_point) < alignment_tolerance:  # Allowable tolerance
                    print("Angles aligned.")
                    break
                else:
                    print("Angles not aligned. Re-aligning...")
                    retries += 1

            else:
                print("Error fetching updated instructions")
                robot.straight(-100)
                print("Backing up because max_retries reached")
                break

        if retries >= max_retries:
            print("Max retries reached, proceeding without perfect alignment")
    except Exception as e:
        print("Error in align_robot:", e)

# Function to start the motors at ports A and D
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

# Function to stop the motors at ports A and D
def stop_motors(motor_a, motor_d):
    try:
        motor_a.stop()
        motor_d.stop()
        print("Motors stopped")
    except Exception as e:
        print("Error in stop_motors:", e)

def main():
    try:
        # Objects and setup
        ev3 = EV3Brick()
        left_wheel = Motor(Port.C)
        right_wheel = Motor(Port.B)
        motor_a = Motor(Port.A, Direction.COUNTERCLOCKWISE)
        motor_d = Motor(Port.D, Direction.CLOCKWISE)

        # Wheel diameter and axle track (in millimeters)
        wheel_diameter = 54.5
        axle_track = 180

        # DriveBase object
        robot = DriveBase(left_wheel, right_wheel, wheel_diameter, axle_track)

        # Define the server URL
        server_url = "http://172.20.10.2:5002"  # Replace <server-ip> with the actual server IP address

        # Continuously fetch instructions from the server until successful
        instructions = None
        while instructions is None:
            print("Fetching instructions from the server...")
            instructions = fetch_instructions(server_url)
            if instructions is None:
                print("Retrying to fetch instructions...")
                time.sleep(5)  # Wait before retrying

        # Execute the fetched instructions
        while True:
            if instructions["action"] == "move_to_ball":
                ball_point = instructions["ball_point"]
                blue_center = (instructions["blue_center"]["x"], instructions["blue_center"]["y"])
                green_center = (instructions["green_center"]["x"], instructions["green_center"]["y"])
                half_distance_cm = instructions["half_distance_cm"]
                remaining_distance_cm = instructions["remaining_distance_cm"]
                full_distance_cm = instructions["full_distance_cm"]
                safe_point = instructions["safe_point"]

                print("Initial instructions: ball_point={}, blue_center={}, green_center={}, half_distance_cm={}, remaining_distance_cm={}, full_distance_cm={}, safe_point={}".format(ball_point, blue_center, green_center, half_distance_cm, remaining_distance_cm, full_distance_cm, safe_point))

                # Align the robot with the ball
                align_robot(robot, blue_center, green_center, ball_point, correction_factor, server_url)

                if full_distance_cm > 20:
                    # Move half the distance to the ball if the full distance is greater than 20 cm
                    move_to_point(robot, half_distance_cm, motor_a, motor_d)
                    print("Moved half the distance to ball")

                    # Fetch updated instructions for the remaining distance
                    instructions = fetch_instructions(server_url)
                    if instructions and instructions["action"] == "move_to_ball":
                        ball_point = instructions["ball_point"]
                        blue_center = (instructions["blue_center"]["x"], instructions["blue_center"]["y"])
                        green_center = (instructions["green_center"]["x"], instructions["green_center"]["y"])
                        remaining_distance_cm = remaining_distance_cm

                        print("Updated instructions: ball_point={}, blue_center={}, green_center={}".format(ball_point, blue_center, green_center))

                        # Re-align the robot with the ball
                        align_robot(robot, blue_center, green_center, ball_point, correction_factor, server_url)

                        # Move the remaining distance to the ball
                        move_to_point(robot, remaining_distance_cm, motor_a, motor_d)
                        print("Moved remaining distance to ball")
                    else:
                        print("Received idle instruction or no ball detected")
                        print(instructions["message"] if instructions else "No instructions received")
                else:
                    # Move directly to the ball if the full distance is 20 cm or less
                    move_to_point(robot, full_distance_cm, motor_a, motor_d)
                    print("Moved full distance to ball")

                # Move to safe point after picking up the ball
                align_robot(robot, blue_center, green_center, safe_point, correction_factor, server_url)
                move_to_point(robot, remaining_distance_cm, motor_a, motor_d)
                print("Moved to safe point")

            elif instructions["action"] == "move_to_dot":
                dot_point = instructions["small_goal_dot"]
                blue_center = (instructions["blue_center"]["x"], instructions["blue_center"]["y"])
                green_center = (instructions["green_center"]["x"], instructions["green_center"]["y"])
                half_distance_cm = instructions["half_distance_cm"]
                remaining_distance_cm = instructions["remaining_distance_cm"]
                full_distance_cm = instructions["full_distance_cm"]
                small_goal = instructions["small_goal"]
                safe_point = instructions["safe_point"]

                print("Initial instructions: dot_point={}, blue_center={}, green_center={}, half_distance_cm={}, remaining_distance_cm={}, full_distance_cm={}, small_goal={}, safe_point={}".format(dot_point, blue_center, green_center, half_distance_cm, remaining_distance_cm, full_distance_cm, small_goal, safe_point))

                # Align the robot with the dot
                align_robot(robot, blue_center, green_center, dot_point, correction_factor, server_url)
                print("The robot is aligned moving to if full_distsance")

                if full_distance_cm > 20:
                    # Move half the distance to the dot if the full distance is greater than 20 cm
                    print("Full disntance over 20 cm, move to dot function")
                    move_to_dot_function(robot, half_distance_cm)
                    print("Moved half the distance to dot")

                    # Fetch updated instructions for the remaining distance
                    instructions = fetch_instructions(server_url)
                    if instructions and instructions["action"] == "move_to_dot":
                        dot_point = instructions["small_goal_dot"]
                        blue_center = (instructions["blue_center"]["x"], instructions["blue_center"]["y"])
                        green_center = (instructions["green_center"]["x"], instructions["green_center"]["y"])
                        remaining_distance_cm = instructions["remaining_distance_cm"]

                        print("Updated instructions: dot_point={}, blue_center={}, green_center={}".format(dot_point, blue_center, green_center))

                        # Re-align the robot with the dot
                        align_robot(robot, blue_center, green_center, dot_point, correction_factor, server_url)

                        # Move the remaining distance to the dot
                        move_to_dot_function(robot, remaining_distance_cm)
                        print("Moved remaining distance to dot, trying to align")
                        align_robot(robot, blue_center, green_center, small_goal, correction_factor, server_url)
                        robot.straight(100)
                        time.sleep(2)
                        print("Moved to small goal")

                    else:
                        move_to_dot_function(robot, remaining_distance_cm)
                        print("Received idle instruction or no dot detected")
                        robot.straight(100)
                        print(instructions["message"] if instructions else "No instructions received")
                else:
                    # Move directly to the dot if the full distance is 20 cm or less
                    move_to_dot_function(robot, full_distance_cm)
                    print("Moved full distance to dot")
                    align_robot(robot, blue_center, green_center, small_goal, correction_factor, server_url)
                    robot.straight(100)
                    time.sleep(2)
                    print("Moved to small goal")

                # Start motors when in front of the dot and run for a specified duration
                instructions = fetch_instructions(server_url)
                run_time = 6  # Run motors for 6 seconds (adjust as needed)
                start_motors_reverse(motor_a, motor_d)
                time.sleep(run_time)
                stop_motors(motor_a, motor_d)

                # Break the loop after moving to the dot
                break

            elif instructions["action"] == "move_to_ball_close_to_field":
                ball_point = instructions["ball_point"]
                field_dot = instructions["field_dot"]
                out_dot = instructions["out_dot"]
                blue_center = (instructions["blue_center"]["x"], instructions["blue_center"]["y"])
                green_center = (instructions["green_center"]["x"], instructions["green_center"]["y"])
                distance_to_ball = instructions["full_distance_cm"]
                remaining_distance_cm = instructions["remaining_distance_cm"]
                safe_point = instructions["safe_point"]

                print("Initial instructions: ball_point={}, field_dot={}, out_dot={}, blue_center={}, green_center={}, distance_to_ball={}, safe_point={}".format(ball_point, field_dot, out_dot, blue_center, green_center, distance_to_ball, safe_point))

                # Align the robot with the ball close to the field
                align_robot(robot, blue_center, green_center, out_dot, correction_factor, server_url)

                if full_distance_cm > 20:
                    # Move half the distance to the dot if the full distance is greater than 20 cm
                    print("Full disntance over 20 cm, move to out dot function")
                    move_to_dot_function(robot, half_distance_cm)
                    print("Moved half the distance to out dot")

                    # Fetch updated instructions for the remaining distance
                    instructions = fetch_instructions(server_url)
                    if instructions["action"] == "move_to_ball_close_to_field":
                        ball_point = instructions["ball_point"]
                        field_dot = instructions["field_dot"]
                        out_dot = instructions["out_dot"]
                        blue_center = (instructions["blue_center"]["x"], instructions["blue_center"]["y"])
                        green_center = (instructions["green_center"]["x"], instructions["green_center"]["y"])
                        distance_to_ball = instructions["full_distance_cm"]
                        remaining_distance_cm = instructions["remaining_distance_cm"]

                        print("Updated instructions: dot_point={}, blue_center={}, green_center={}".format(dot_point, blue_center, green_center))

                        # Re-align the robot with the dot
                        align_robot(robot, blue_center, green_center, out_dot, correction_factor, server_url)

                        # Move the remaining distance to the dot
                        move_to_dot_function(robot, remaining_distance_cm)
                        print("Moved remaining distance to dot, trying to align")
                        align_robot(robot, blue_center, green_center, ball_point, correction_factor, server_url)
                        robot.straight(10)
                        time.sleep(2)
                        start_motors(motor_a, motor_d)
                        time.sleep(5)
                        print("backing in cased we fucked up")
                        robot.straight(-100)
                    else:
                        move_to_dot_function(robot, remaining_distance_cm)
                        robot.straight(10)
                        time.sleep(2)
                        start_motors(motor_a, motor_d)
                        time.sleep(5)
                        robot.straight(-100)
                        print(instructions["message"] if instructions else "No instructions received")
                else:
                    # Move directly to the dot if the full distance is 20 cm or less
                    move_to_dot_function(robot, full_distance_cm)
                    robot.straight(10)
                    time.sleep(2)
                    start_motors(motor_a, motor_d)
                    time.sleep(5)
                    robot.straight(-100)
                    print("Moved to small goal")
                    print("Moved full distance to dot")
                    align_robot(robot, blue_center, green_center, small_goal, correction_factor, server_url)
                    robot.straight(100)
                    time.sleep(2)
                    print("Moved to small goal")
            else:
                print("Received idle instruction or no ball detected")
                print(instructions["message"] if instructions else "No instructions received")

            # Fetch updated instructions
            instructions = fetch_instructions(server_url)
            if instructions is None:
                print("Retrying to fetch instructions...")
                time.sleep(5)  # Wait before retrying
                instructions = fetch_instructions(server_url)

        # Wait for a while to observe the results
        print("Done!")
        time.sleep(2)

    except Exception as e:
        print("Error in main:", e)

if __name__ == "__main__":
    main()
