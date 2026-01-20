import cv2
import mediapipe as mp
import keyboard 
from time import sleep



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

        raised_left = 0
        raised_right = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                label = handedness.classification[0].label  # "Left" of "Right"

                if is_hand_raised(hand_landmarks):
                    if label == "Right":
                        raised_right += 1
                    elif label == "Left":
                        raised_left += 1

        # Twee rechter handen omhoog → "a"
        if raised_right >= 2 and not condition_triggered:
            print("Twee rechter handen omhoog → toets 'a'")
            keyboard.press_and_release("a")
            condition_triggered = True
            # sleep(3)  # korte pauze om meerdere triggers te voorkomen

        # Twee linker handen omhoog → "b"
        elif raised_left >= 2 and not condition_triggered:
            print("Twee linker handen omhoog → toets 'b'")
            keyboard.press_and_release("b")
            condition_triggered = True

        # Twee verschillende handen gezien (1 links + 1 rechts omhoog)
        if raised_left == 1 and raised_right == 1 and not condition_triggered:
            print("Er worden twee verschillende handen gezien!")
            keyboard.press_and_release("right")
            sleep(2)  # korte pauze voor stabiliteit
            # sleep(5)  # korte pauze om meerdere triggers te voorkomen
            condition_triggered = True

        

        # Reset als geen van beide condities actief is
        if raised_right < 2 and raised_left < 2:
            condition_triggered = False

        cv2.imshow("Hand detectie", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

