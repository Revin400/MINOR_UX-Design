import cv2
import mediapipe as mp
import keyboard 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

condition_triggered = False

def is_hand_raised(hand_landmarks):
    wrist_y = hand_landmarks.landmark[0].y
    fingertip_ids = [4, 8, 12, 16, 20]
    fingertip_ys = [hand_landmarks.landmark[i].y for i in fingertip_ids]
    avg_fingertip_y = sum(fingertip_ys) / len(fingertip_ys)
    return avg_fingertip_y < wrist_y - 0.05

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        raised_hands = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                if is_hand_raised(hand_landmarks):
                    raised_hands += 1

        # KEY PRESS BIJ TWEE HANDEN OMHOOG
        if raised_hands >= 2 and not condition_triggered:
            print("Beide handen omhoog → druk op 'd'")
            # keyboard.press_and_release("d")   # ← HIER GEBEURT DE KEYPRESS
            keyboard.press_and_release("right")

            condition_triggered = True

        if raised_hands < 2:
            condition_triggered = False  # reset zodat je opnieuw kan triggeren

        cv2.imshow("Hand detectie", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
