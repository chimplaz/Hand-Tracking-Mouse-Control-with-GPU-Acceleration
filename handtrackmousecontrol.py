import cv2
import mediapipe as mp
import pyautogui
import cupy as cp
import numpy as np
import time

# Initialize Mediapipe and PyAutoGUI
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Lowering the confidence
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# CUDA-accelerated smoothing function using CuPy
def smooth_movement(prev_pos, curr_pos, alpha=0.7):
    prev_cp = cp.array(prev_pos, dtype=cp.float32)  # Move to GPU
    curr_cp = cp.array(curr_pos, dtype=cp.float32)
    smoothed_cp = alpha * prev_cp + (1 - alpha) * curr_cp
    return smoothed_cp.get()  # Bring the result back to CPU

# Get GPU name using CuPy
device_id = 0  # Default GPU device
gpu_properties = cp.cuda.runtime.getDeviceProperties(device_id)
gpu_name = gpu_properties['name'].decode('utf-8')

# Webcam capture
cap = cv2.VideoCapture(0)
prev_position = (0, 0)
clicking = False
prev_finger_distance = 0
scrolling = False
prev_scroll_position = None

# FPS calculation
prev_time = 0
fps = 0

print(f"Using GPU: {gpu_name}")
print("Starting hand-tracking mouse control. Press 'q' to exit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting...")
            break

        # Resize the frame to increase FPS (reduce resolution for faster processing)
        frame_resized = cv2.resize(frame, (640, 480))  # Resize to a lower resolution
        frame_height, frame_width, _ = frame_resized.shape

        # Convert BGR frame to RGB
        rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get coordinates of the index finger tip (landmark 8)
                x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width
                y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height

                # Get coordinates of the thumb tip (landmark 4)
                thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame_width
                thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame_height

                # Distance between index finger and thumb (used for clicking)
                finger_distance = np.linalg.norm([x - thumb_x, y - thumb_y])

                # Map coordinates to screen size using GPU
                screen_x = np.interp(x, [0, frame_width], [0, screen_width])
                screen_y = np.interp(y, [0, frame_height], [0, screen_height])

                # Smooth movement using GPU-accelerated function
                smoothed_position = smooth_movement(prev_position, (screen_x, screen_y))
                prev_position = smoothed_position

                # Move the mouse
                pyautogui.moveTo(smoothed_position[0], smoothed_position[1])

                # Check for click gesture (close finger distance)
                if finger_distance < 30 and not clicking:  # Finger close -> click
                    pyautogui.click()
                    clicking = True
                elif finger_distance >= 30:
                    clicking = False  # Reset click state

                # Scroll up/down based on the vertical movement of the hand
                if len(hand_landmarks.landmark) >= 21:
                    # Get distance between index finger and middle finger
                    index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    middle_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                    scroll_distance = index_finger_y - middle_finger_y

                    if abs(scroll_distance) > 0.05:  # Threshold for scrolling
                        if scroll_distance > 0:  # Scroll up
                            pyautogui.scroll(10)
                        else:  # Scroll down
                            pyautogui.scroll(-10)

                # Draw the landmarks on the frame
                mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Display FPS and GPU name on the frame
        cv2.putText(frame_resized, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_resized, f"GPU: {gpu_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow("Hand Tracking Mouse Control", frame_resized)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Program exited.")
