import cv2
import mediapipe as mp
import time
import math
import mouse   # <-- NIEUW

print("Pinch zoom detector - spread fingers => zoom in (scroll up), pinch => unzoom (scroll down)")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# Instellingen
# =========================
ZOOM_IN_THRESHOLD  = 0.085
UNZOOM_THRESHOLD   = 0.060
EVENT_COOLDOWN_SEC = 0.25

SCROLL_AMOUNT = 5   # hoe sterk er gescrold wordt per trigger

def dist_2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    is_zoomed = False
    last_event_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pinch_dist = None

        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            thumb_tip = (hl.landmark[4].x, hl.landmark[4].y)
            index_tip = (hl.landmark[8].x, hl.landmark[8].y)

            pinch_dist = dist_2d(thumb_tip, index_tip)

            # Visual feedback
            tpx, tpy = int(thumb_tip[0] * w), int(thumb_tip[1] * h)
            ipx, ipy = int(index_tip[0] * w), int(index_tip[1] * h)
            cv2.circle(frame, (tpx, tpy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (ipx, ipy), 10, (255, 255, 255), -1)
            cv2.line(frame, (tpx, tpy), (ipx, ipy), (255, 255, 255), 2)

            now = time.time()
            can_fire = (now - last_event_t) >= EVENT_COOLDOWN_SEC

            if can_fire:
                # ===== ZOOM IN =====
                if (not is_zoomed) and pinch_dist >= ZOOM_IN_THRESHOLD:
                    print("zoom in")
                    mouse.wheel(SCROLL_AMOUNT)   # <-- scroll up
                    is_zoomed = True
                    last_event_t = now

                # ===== ZOOM OUT =====
                elif is_zoomed and pinch_dist <= UNZOOM_THRESHOLD:
                    print("unzoom")
                    mouse.wheel(-SCROLL_AMOUNT)  # <-- scroll down
                    is_zoomed = False
                    last_event_t = now

        # UI
        status = "ZOOMED" if is_zoomed else "UNZOOMED"
        cv2.putText(frame, f"Status: {status}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        if pinch_dist is not None:
            cv2.putText(frame, f"Pinch dist: {pinch_dist:.3f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Pinch Zoom Scroll", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
