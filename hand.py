import cv2
# import keyboard
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Webcam openen (0 = default camera)
cap = cv2.VideoCapture(0)

# Flags
condition_triggered = False  # zodat we maar één keer "door" geven

def is_hand_raised(hand_landmarks):
    """
    Eenvoudige check: is de gemiddelde y van de vingertoppen hoger (kleiner y) dan de pols?
    In beeldcoördinaten is boven = kleinere y.
    """
    wrist_y = hand_landmarks.landmark[0].y  # pols
    fingertip_ids = [4, 8, 12, 16, 20]      # duim, wijs, middel, ring, pink top

    fingertip_ys = [hand_landmarks.landmark[i].y for i in fingertip_ids]
    avg_fingertip_y = sum(fingertip_ys) / len(fingertip_ys)

    # Als vingers duidelijk boven de pols zitten -> hand omhoog
    return avg_fingertip_y < wrist_y - 0.05  # marge om wat tolerant te zijn

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
            print("Kon geen frame van de camera lezen.")
            break

        # Spiegelbeeld maken zodat het natuurlijker voelt
        frame = cv2.flip(frame, 1)

        # BGR -> RGB voor Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        raised_hands = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                label = handedness.classification[0].label  # "Left" of "Right"

                # Tekenen op het beeld
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Check of deze hand omhoog is
                if is_hand_raised(hand_landmarks):
                    raised_hands += 1
                    cv2.putText(
                        frame,
                        f"{label} UP",
                        (10, 30 * raised_hands),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # Als minstens twee handen omhoog zijn -> systeem mag door
        # if raised_hands >= 2 and not condition_triggered:
        #     print("✅ Beide handen omhoog, systeem mag door! (links en rechts)")

        #     condition_triggered = True 

        # Info overlay
        cv2.putText(
            frame,
            "Steek twee handen omhoog (q om te stoppen)",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Hand detectie", frame)

        # Stoppen met 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Netjes afsluiten
cap.release()
cv2.destroyAllWindows()
