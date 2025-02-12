import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip position (landmark 8)
            index_finger = hand_landmarks.landmark[8]
            x, y = int(index_finger.x * screen_w), int(index_finger.y * screen_h)

            # Move mouse pointer smoothly
            pyautogui.moveTo(x, y, duration=0.1)

            # Get thumb position (landmark 4)
            thumb = hand_landmarks.landmark[4]
            thumb_x, thumb_y = int(thumb.x * screen_w), int(thumb.y * screen_h)

            # Detect click when thumb & index finger are close
            distance = np.sqrt((thumb_x - x)**2 + (thumb_y - y)**2)
            if distance < 40:  # Click threshold
                pyautogui.click()
                pyautogui.sleep(0.2)  # Prevent rapid multiple clicks

            # Get middle finger tip position (landmark 12)
            middle_finger = hand_landmarks.landmark[12]
            middle_x, middle_y = int(middle_finger.x * screen_w), int(middle_finger.y * screen_h)

            # Scroll up/down based on middle finger position
            if middle_y < y:
                pyautogui.scroll(10)  # Scroll up
            elif middle_y > y:
                pyautogui.scroll(-10)  # Scroll down

    cv2.imshow("Virtual Mouse (Click & Scroll)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
