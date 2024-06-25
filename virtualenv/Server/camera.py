import cv2
import numpy as np
from flask import Flask, jsonify, Response
from ultralytics import YOLO
import math
import time

app = Flask(__name__)

# Load the exported OpenVINO model
ov_model = YOLO("/Users/ahmadhaj/Desktop/CDIOProjekt/virtualenv/Server/Assets/test6_openvino_model")

# Initialize video capture (0 for default webcam, or provide video file path)
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

# Global variables to store the small goal's center and other data
small_goal_center_global = None
balls_not_close_to_field_global = []
conversion_factor_global = None  # Global conversion factor
conversion_factor_set = False  # Flag to indicate if the conversion factor is set

no_more_balls_in_open = 0  # Counter for no more balls in open
last_ball_check_time = None  # Timestamp for the last check

def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.equalizeHist(v)
    hsv = cv2.merge([h, s, v])
    return hsv

def detect_green(frame):
    hsv = preprocess_frame(frame)
    lower_green = np.array([20, 50, 100])  # Adjusted lower bound
    upper_green = np.array([90, 255, 255])  # Adjusted upper bound
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x + w // 2, y + h // 2)
    return None

def detect_blue(frame):
    hsv = preprocess_frame(frame)
    lower_blue = np.array([50, 110, 155])
    upper_blue = np.array([120, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x + w // 2, y + h // 2)
    return None

# Function to detect field
def detect_field(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 102, 98])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([170, 102, 98])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological transformations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_contour = []
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(contour) > 1:
            rectangle_contour.append(contour)
    if rectangle_contour:
        largest_rectangle = max(rectangle_contour, key=cv2.contourArea)
        cv2.drawContours(frame, [largest_rectangle], -1, (255, 255, 255), 3)
        return largest_rectangle, contours
    else:
        return None, contours

# Function to calculate the center of the bounding box
def get_center(bbox):
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return center_x, center_y

# Function to calculate the conversion factor
def calculate_conversion_factor(rectangle):
    real_width = 200  # cm
    real_height = 130  # cm
    rect = cv2.minAreaRect(rectangle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width_pixels = max(rect[1][0], rect[1][1])
    height_pixels = min(rect[1][0], rect[1][1])
    conversion_factor = (real_width / width_pixels + real_height / height_pixels) / 2
    return conversion_factor

# Function to calculate the real-world distance using the conversion factor
def calculate_real_distance(observed_distance_px, conversion_factor):
    if conversion_factor is None:
        raise ValueError("Conversion factor has not been set.")
    return observed_distance_px * conversion_factor

# Function to calculate the angle between two points
def calculate_angle(center1, center2):
    x1, y1 = center1
    x2, y2 = center2
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return angle

# Function to find the closest ball to the green object
def find_closest_ball(green_center, balls_centers, conversion_factor):
    if not green_center or not balls_centers:
        return None, float('inf')
    closest_ball = None
    min_distance = float('inf')
    for ball_center in balls_centers:
        observed_distance_px = np.sqrt((green_center[0] - ball_center[0]) ** 2 + (green_center[1] - ball_center[1]) ** 2)
        real_distance = calculate_real_distance(observed_distance_px, conversion_factor)
        if real_distance < min_distance:
            min_distance = real_distance
            closest_ball = ball_center
    return closest_ball, min_distance

# Function to check if a ball is close to the field boundary
def is_ball_close_to_field(ball_center, field_contours, conversion_factor):
    if conversion_factor is None:
        print("Error: Conversion factor is None when checking if ball is close to field.")
        return False
    for contour in field_contours:
        for point in contour:
            point = point[0]
            observed_distance_px = np.sqrt((ball_center[0] - point[0]) ** 2 + (ball_center[1] - point[1]) ** 2)
            real_distance = calculate_real_distance(observed_distance_px, conversion_factor)
            if real_distance < 12:  # 12 cm threshold
                return True
    return False

# Function to generate dots for picking up balls close to the field boundary
def generate_field_dots(ball_center, field_contours, conversion_factor):
    if conversion_factor is None:
        print("Error: Conversion factor is None when generating field dots.")
        return None, None
    for contour in field_contours:
        for point in contour:
            point = point[0]
            observed_distance_px = np.sqrt((ball_center[0] - point[0]) ** 2 + (ball_center[1] - point[1]) ** 2)
            real_distance = calculate_real_distance(observed_distance_px, conversion_factor)
            if real_distance < 12:
                field_dot = (point[0], point[1])
                out_dot = (ball_center[0] + int(20 / conversion_factor * math.cos(math.radians(0))),
                           ball_center[1] - int(20 / conversion_factor * math.sin(math.radians(0))))
                return field_dot, out_dot
    return None, None

# Function to generate a dot for the small goal
def generate_small_goal_dot(small_goal_center, conversion_factor):
    if small_goal_center and conversion_factor:
        angle = 0  # Assuming the dot is directly in front of the goal, you may need to adjust this angle as per your setup
        distance_px = int(20 / conversion_factor)
        small_goal_dot = (
            small_goal_center[0] + distance_px,
            small_goal_center[1]
        )
        return small_goal_dot
    return None

def generate_frame():
    global small_goal_center_global, balls_not_close_to_field_global, conversion_factor_global, conversion_factor_set, no_more_balls_in_open, last_ball_check_time
    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        frame_height, frame_width = frame.shape[:2]

        green_center = detect_green(frame)
        blue_center = detect_blue(frame)
        field_rectangle, field_contours = detect_field(frame)

        if green_center:
            print(f"Green center detected at: {green_center}")
        else:
            print("Green center not detected.")

        if blue_center:
            print(f"Blue center detected at: {blue_center}")
        else:
            print("Blue center not detected.")


        if field_rectangle is not None and not conversion_factor_set:
            print("Field rectangle detected.")
            conversion_factor_global = calculate_conversion_factor(field_rectangle)
            conversion_factor_set = True
            print(f"Conversion factor: {conversion_factor_global}")

        if conversion_factor_global is None:
            print("Conversion factor not set, skipping frame.")
            continue

        results = ov_model(frame, conf=0.4, imgsz=640)

        ball_centers = []
        small_goal_center = None
        small_goal_confidence = 0

        for result in results:
            for detection in result.boxes.data.cpu().numpy():
                bbox = detection[:4]
                confidence = detection[4]
                class_id = int(detection[5])
                label = ov_model.names[class_id]

                if confidence > 0.4:
                    center = get_center(bbox)
                    if label == "white ball" or label == "orange ball":
                        ball_centers.append(center)
                    elif label == "small goal":
                        if confidence > 0.4 and confidence > small_goal_confidence and center[0] < 800:
                            small_goal_center = center
                            small_goal_confidence = confidence

                    if label not in ["walls", "robot green", "robot black", "big goal"]:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        label_text = f"{label}: {confidence:.2f}"
                        cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        balls_not_close_to_field = []
        balls_close_to_field = []

        for ball_center in ball_centers:
            if is_ball_close_to_field(ball_center, field_contours, conversion_factor_global):
                balls_close_to_field.append(ball_center)
            else:
                balls_not_close_to_field.append(ball_center)

        if balls_not_close_to_field_global == []:
            balls_not_close_to_field_global = balls_not_close_to_field

        # Save the small goal center if detected
        if small_goal_center and small_goal_center_global is None:
            small_goal_center_global = small_goal_center

        # Generate the small goal dot using the global small goal center
        small_goal_dot_json = {}
        small_goal_dot = None
        if small_goal_center_global:
            small_goal_dot = generate_small_goal_dot(small_goal_center_global, conversion_factor_global)
            if small_goal_dot:
                small_goal_dot_json = {
                    "small_goal_dot": {
                        "x": int(small_goal_dot[0]),
                        "y": int(small_goal_dot[1])
                    }
                }
                # Annotate the small goal and the dot
                cv2.circle(frame, small_goal_center_global, 5, (0, 0, 255), -1)
                cv2.circle(frame, small_goal_dot, 5, (0, 255, 255), -1)
                cv2.putText(frame, f"Small Goal: {small_goal_confidence:.2f}", (small_goal_center_global[0] + 10, small_goal_center_global[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, f"Dot: {small_goal_dot}", (small_goal_dot[0] + 10, small_goal_dot[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Check for balls in the open
        if balls_not_close_to_field:
            no_more_balls_in_open = 0
            last_ball_check_time = None
        else:
            if last_ball_check_time is None:
                last_ball_check_time = time.time()
            elif time.time() - last_ball_check_time > 10:
                no_more_balls_in_open += 1
                last_ball_check_time = None

        if green_center:
            for ball_center in balls_not_close_to_field:
                angle_to_ball = calculate_angle(green_center, ball_center)
                distance_to_ball = calculate_real_distance(np.sqrt((green_center[0] - ball_center[0]) ** 2 + (green_center[1] - ball_center[1]) ** 2), conversion_factor_global)
                cv2.line(frame, green_center, ball_center, (0, 255, 0), 2)
                cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Angle: {angle_to_ball:.2f}", (ball_center[0] + 10, ball_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Distance: {distance_to_ball:.2f} cm", (ball_center[0] + 10, ball_center[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if blue_center:
                    cv2.line(frame, blue_center, ball_center, (255, 0, 0), 2)

            for ball_center in balls_close_to_field:
                field_dot, out_dot = generate_field_dots(ball_center, field_contours, conversion_factor_global)
                if field_dot and out_dot:
                    angle_to_ball = calculate_angle(out_dot, ball_center)
                    distance_to_ball = calculate_real_distance(np.sqrt((out_dot[0] - ball_center[0]) ** 2 + (out_dot[1] - ball_center[1]) ** 2), conversion_factor_global)
                    cv2.line(frame, out_dot, ball_center, (0, 255, 0), 2)
                    cv2.circle(frame, ball_center, 5, (0, 0, 255), -1)
                    cv2.circle(frame, out_dot, 5, (255, 0, 0), -1)
                    cv2.circle(frame, field_dot, 5, (255, 255, 0), -1)
                    cv2.putText(frame, f"Angle: {angle_to_ball:.2f}", (ball_center[0] + 10, ball_center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Distance: {distance_to_ball:.2f} cm", (ball_center[0] + 10, ball_center[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if blue_center:
                        cv2.line(frame, blue_center, ball_center, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()  # Use tobytes() instead of to_bytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    global small_goal_center_global, balls_not_close_to_field_global, conversion_factor_global, conversion_factor_set, no_more_balls_in_open, last_ball_check_time
    retry_count = 0
    max_retries = 15
    while retry_count < max_retries:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Warning: Could not read frame. Retrying... ({retry_count + 1}/{max_retries})")
            retry_count += 1
            time.sleep(1)
            continue

        frame_height, frame_width = frame.shape[:2]

        green_center = detect_green(frame)
        blue_center = detect_blue(frame)
        field_rectangle, field_contours = detect_field(frame)

        if green_center:
            print(f"Green center detected at: {green_center}")
        else:
            print("Green center not detected.")

        if blue_center:
            print(f"Blue center detected at: {blue_center}")
        else:
            print("Blue center not detected.")
        
        if field_rectangle is not None and not conversion_factor_set:
            print("Field rectangle detected.")
            conversion_factor_global = calculate_conversion_factor(field_rectangle)
            conversion_factor_set = True
            print(f"Conversion factor: {conversion_factor_global}")
        
        if green_center is None or blue_center is None:
            print(f"Warning: Could not detect robot markers. Retrying... ({retry_count + 1}/{max_retries})")
            retry_count += 1
            time.sleep(1)
            continue

        if conversion_factor_global is None:
            print("Conversion factor not set, skipping frame.")
            continue

        results = ov_model(frame, conf=0.4, imgsz=640)

        ball_centers = []
        small_goal_center = None
        small_goal_confidence = 0
        
        for result in results:
            for detection in result.boxes.data.cpu().numpy():
                bbox = detection[:4]
                confidence = detection[4]
                class_id = int(detection[5])
                label = ov_model.names[class_id]

                if confidence > 0.4:
                    center = get_center(bbox)
                    if label == "white ball" or label == "orange ball":
                        ball_centers.append(center)
                    elif label == "small goal":
                        if confidence > 0.4 and confidence > small_goal_confidence and center[0] < 1000:
                            small_goal_center = center
                            small_goal_confidence = confidence

                    if label not in ["walls", "robot green", "robot black", "big goal"]:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                        label_text = f"{label}: {confidence:.2f}"
                        cv2.putText(frame, label_text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        balls_not_close_to_field = []
        balls_close_to_field = []

        for ball_center in ball_centers:
            if is_ball_close_to_field(ball_center, field_contours, conversion_factor_global):
                balls_close_to_field.append(ball_center)
            else:
                balls_not_close_to_field.append(ball_center)

        if balls_not_close_to_field_global == []:
            balls_not_close_to_field_global = balls_not_close_to_field

        # Save the small goal center if detected
        if small_goal_center and small_goal_center_global is None:
            small_goal_center_global = small_goal_center

        closest_ball, min_distance = find_closest_ball(green_center, balls_not_close_to_field_global, conversion_factor_global)

        small_goal_dot_json = {}
        small_goal_dot = None
        if small_goal_center_global:
            small_goal_dot = generate_small_goal_dot(small_goal_center_global, conversion_factor_global)
            if small_goal_dot:
                small_goal_dot_json = {
                    "small_goal_dot": {
                        "x": int(small_goal_dot[0]),
                        "y": int(small_goal_dot[1])
                    }
                }

        # Check for balls in the open
        if balls_not_close_to_field:
            no_more_balls_in_open = 0
            last_ball_check_time = None
        else:
            if last_ball_check_time is None:
                last_ball_check_time = time.time()
            elif time.time() - last_ball_check_time > 10:
                no_more_balls_in_open += 1
                last_ball_check_time = None

        if closest_ball:
            rotation_angle = 0
            if green_center:
                angle_green_to_ball = calculate_angle(green_center, closest_ball)
                cv2.line(frame, green_center, closest_ball, (0, 255, 0), 2)
                cv2.circle(frame, green_center, 5, (0, 255, 0), -1)
                rotation_angle = angle_green_to_ball
                cv2.putText(frame, f"Green to Ball Angle: {angle_green_to_ball:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Green: {green_center}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if blue_center:
                angle_blue_to_ball = calculate_angle(blue_center, closest_ball)
                cv2.line(frame, blue_center, closest_ball, (255, 0, 0), 2)
                cv2.circle(frame, blue_center, 5, (255, 0, 255), -1)
                rotation_angle -= angle_blue_to_ball
                cv2.putText(frame, f"Blue to Ball Angle: {angle_blue_to_ball:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"Blue: {blue_center}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.circle(frame, closest_ball, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Distance: {min_distance:.2f} cm", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Ball: {closest_ball}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            half_distance_cm = min_distance / 2
            remaining_distance_cm = min_distance - half_distance_cm

            instruction = {
                "action": "move_to_ball",
                "ball_point": {
                    "x": int(closest_ball[0]),
                    "y": int(closest_ball[1])
                },
                "half_distance_cm": half_distance_cm,
                "remaining_distance_cm": remaining_distance_cm,
                "full_distance_cm": min_distance,
                "angle": rotation_angle,
                "blue_center": {
                    "x": int(blue_center[0]),
                    "y": int(blue_center[1])
                },
                "green_center": {
                    "x": int(green_center[0]),
                    "y": int(green_center[1])
                }
            }

            # Remove the picked-up ball from the global list
            balls_not_close_to_field_global.remove(closest_ball)
        elif balls_close_to_field and no_more_balls_in_open > 0:
            for closest_ball in balls_close_to_field:
                field_dot, out_dot = generate_field_dots(closest_ball, field_contours, conversion_factor_global)
                if field_dot and out_dot:
                    angle_to_ball = calculate_angle(out_dot, closest_ball)
                    distance_to_ball = calculate_real_distance(np.sqrt((out_dot[0] - closest_ball[0]) ** 2 + (out_dot[1] - closest_ball[1]) ** 2), conversion_factor_global)
                    rotation_angle = angle_to_ball

                    instruction = {
                        "action": "move_to_ball_close_to_field",
                        "field_dot": {
                            "x": int(field_dot[0]),
                            "y": int(field_dot[1])
                        },
                        "out_dot": {
                            "x": int(out_dot[0]),
                            "y": int(out_dot[1])
                        },
                        "ball_point": {
                            "x": int(closest_ball[0]),
                            "y": int(closest_ball[1])
                        },
                        "full_distance_cm": distance_to_ball,
                        "angle": rotation_angle,
                        "blue_center": {
                            "x": int(blue_center[0]),
                            "y": int(blue_center[1])
                        },
                        "green_center": {
                            "x": int(green_center[0]),
                            "y": int(green_center[1])
                        },
                        "half_distance_cm": distance_to_ball / 2,
                        "remaining_distance_cm": distance_to_ball
            
                    }
                    break

        else:
            if green_center and small_goal_dot:
                distance_to_dot = calculate_real_distance(np.sqrt((green_center[0] - small_goal_dot[0]) ** 2 + (green_center[1] - small_goal_dot[1]) ** 2), conversion_factor_global)
            else:
                distance_to_dot = 0
            
            instruction = {
                "action": "move_to_dot",
                "dot_point": {
                    "x": int(small_goal_dot[0]),
                    "y": int(small_goal_dot[1])
                },
                "blue_center": {
                    "x": int(blue_center[0]),
                    "y": int(blue_center[1])
                },
                "green_center": {
                    "x": int(green_center[0]),
                    "y": int(green_center[1])
                },
                "half_distance_cm": distance_to_dot / 2,
                "remaining_distance_cm": distance_to_dot / 2,
                "full_distance_cm": distance_to_dot,
                "small_goal": {
                    "x": int(small_goal_center_global[0]),
                    "y": int(small_goal_center_global[1])
                }
            }

        response = instruction
        if small_goal_dot:
            response.update(small_goal_dot_json)

        return jsonify(response)

    return jsonify({"error": "Max retries exceeded"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)


