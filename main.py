import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

drawing_canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White canvas

drawing_color = (0, 0, 255)
brush_thickness = 5

previous_points = []
drawing_active = False


def count_extended_fingers(landmarks, h, w):
    extended_fingers = 0

    thumb_base = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]

    if thumb_base.y > thumb_mcp.y:
        extended_fingers += 1

    if landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks.landmark[
        mp_hands.HandLandmark.INDEX_FINGER_MCP].y:
        extended_fingers += 1

    if landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks.landmark[
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y:
        extended_fingers += 1

    if landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks.landmark[
        mp_hands.HandLandmark.RING_FINGER_MCP].y:
        extended_fingers += 1

    if landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y:
        extended_fingers += 1

    return extended_fingers


def draw_smooth_line(prev_point, current_point, canvas, color, thickness):
    if prev_point is not None:
        cv2.line(canvas, prev_point, current_point, color, thickness)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            extended_fingers = count_extended_fingers(landmarks, frame.shape[0], frame.shape[1])

            finger_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_base = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            h, w, _ = frame.shape
            tip_x, tip_y = int(finger_tip.x * w), int(finger_tip.y * h)
            base_x, base_y = int(finger_base.x * w), int(finger_base.y * h)

            if extended_fingers == 1:
                if drawing_active:
                    draw_smooth_line(previous_points[-1], (tip_x, tip_y), drawing_canvas, drawing_color,
                                     brush_thickness)
                previous_points.append((tip_x, tip_y))
                drawing_active = True

            elif extended_fingers == 2:
                drawing_active = False
                previous_points.clear()

    cv2.imshow("Camera", frame)

    cv2.imshow("Drawing", drawing_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
