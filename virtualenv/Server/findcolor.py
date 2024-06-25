import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize variables to store the color value
color = None

# Mouse callback function
def get_color(event, x, y, flags, param):
    global color
    if event == cv2.EVENT_LBUTTONDOWN:
        color = frame[y, x]
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Color at ({x}, {y}): {hsv_color}")

        # Adjusting the bounds to better detect red
        if hsv_color[0] < 10 or hsv_color[0] > 170:  # Red color range
            lower_bound1 = np.array([0, max(hsv_color[1] - 100, 50), max(hsv_color[2] - 100, 50)])
            upper_bound1 = np.array([10, 255, 255])
            lower_bound2 = np.array([170, max(hsv_color[1] - 100, 50), max(hsv_color[2] - 100, 50)])
            upper_bound2 = np.array([180, 255, 255])
        else:  # For other colors
            lower_bound1 = np.array([max(hsv_color[0] - 40, 0), max(hsv_color[1] - 100, 50), max(hsv_color[2] - 100, 50)])
            upper_bound1 = np.array([min(hsv_color[0] + 40, 179), 255, 255])
            lower_bound2 = upper_bound2 = None

        print(f"Lower bound 1: {lower_bound1}")
        print(f"Upper bound 1: {upper_bound1}")
        if lower_bound2 is not None and upper_bound2 is not None:
            print(f"Lower bound 2: {lower_bound2}")
            print(f"Upper bound 2: {upper_bound2}")

# Set the mouse callback function
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', get_color)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # If a color is selected, create a mask
    if color is not None:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        if hsv_color[0] < 10 or hsv_color[0] > 170:  # Red color range
            lower_bound1 = np.array([0, max(hsv_color[1] - 100, 50), max(hsv_color[2] - 100, 50)])
            upper_bound1 = np.array([10, 255, 255])
            lower_bound2 = np.array([170, max(hsv_color[1] - 100, 50), max(hsv_color[2] - 100, 50)])
            upper_bound2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv_frame, lower_bound1, upper_bound1)
            mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # For other colors
            lower_bound1 = np.array([max(hsv_color[0] - 40, 0), max(hsv_color[1] - 100, 50), max(hsv_color[2] - 100, 50)])
            upper_bound1 = np.array([min(hsv_color[0] + 40, 179), 255, 255])
            mask = cv2.inRange(hsv_frame, lower_bound1, upper_bound1)
        
        # Apply morphological transformations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply the mask to the frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('Masked', masked_frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
