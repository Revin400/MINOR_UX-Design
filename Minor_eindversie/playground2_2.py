import cv2
import mediapipe as mp
import time
import math
import mouse

print("Smooth continuous pinch-zoom: spread => scroll up, pinch => scroll down (speed depends on distance)")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# TUNE THESE
# =========================
# Neutral distance: waar "geen zoom" is. Zet dit ongeveer tussen jouw open/close.
NEUTRAL_DIST = 0.072

# Deadzone rondom neutral: binnen dit gebied gebeurt er niks (tegen jitter).
DEADZONE = 0.006

# Hoe gevoelig: grotere waarde = minder gevoelig (je moet verder bewegen)
DIST_TO_FULL_SPEED = 0.030  # distance delta die nodig is om max snelheid te halen

# Maximum scroll snelheid (stappen per seconde). Hoger = sneller zoom.
MAX_STEPS_PER_SEC = 25

# Smoothing (0..1): hoger = smoother maar iets meer lag
SMOOTHING_ALPHA = 0.35

# Hand detection
MIN_DET_CONF = 0.7
MIN_TRK_CONF = 0.5

def dist_2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def clamp(x, a, b):
    return max(a, min(b, x))

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRK_CONF
) as hands:

    prev_t = time.time()

    # For smooth speed + wheel accumulator
    smoothed_speed = 0.0         # -1..1 (down..up)
    wheel_accum = 0.0            # accumulates fractional steps

    # For once-per-transition console prints
    last_mode = "idle"           # "zoom_in", "zoom_out", "idle"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        dt = now - prev_t
        prev_t = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        pinch_dist = None
        target_speed = 0.0

        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            # Thumb tip = 4, Index tip = 8
            thumb_tip = (hl.landmark[4].x, hl.landmark[4].y)
            index_tip = (hl.landmark[8].x, hl.landmark[8].y)
            pinch_dist = dist_2d(thumb_tip, index_tip)

            # Visual feedback
            tpx, tpy = int(thumb_tip[0] * w), int(thumb_tip[1] * h)
            ipx, ipy = int(index_tip[0] * w), int(index_tip[1] * h)
            cv2.circle(frame, (tpx, tpy), 10, (255, 255, 255), -1)
            cv2.circle(frame, (ipx, ipy), 10, (255, 255, 255), -1)
            cv2.line(frame, (tpx, tpy), (ipx, ipy), (255, 255, 255), 2)

            # --- Map distance -> speed ---
            delta = pinch_dist - NEUTRAL_DIST

            # Deadzone
            if abs(delta) < DEADZONE:
                target_speed = 0.0
            else:
                # Remove deadzone then normalize to [-1..1]
                if delta > 0:
                    effective = delta - DEADZONE
                else:
                    effective = delta + DEADZONE  # (delta is negative)

                norm = effective / DIST_TO_FULL_SPEED
                target_speed = clamp(norm, -1.0, 1.0)

        else:
            # No hand: gently return to idle (prevents stuck scrolling)
            target_speed = 0.0

        # --- Smooth speed (low-pass filter) ---
        smoothed_speed = (1.0 - SMOOTHING_ALPHA) * smoothed_speed + SMOOTHING_ALPHA * target_speed

        # --- Decide mode + one-time prints on transitions ---
        if smoothed_speed > 0.05:
            mode = "zoom_in"
        elif smoothed_speed < -0.05:
            mode = "zoom_out"
        else:
            mode = "idle"

        if mode != last_mode:
            if mode == "zoom_in":
                print("zoom in (continuous)")
            elif mode == "zoom_out":
                print("zoom out (continuous)")
            else:
                print("idle")
            last_mode = mode

        # --- Continuous scrolling based on speed ---
        # Convert speed (-1..1) to steps/sec
        steps_per_sec = smoothed_speed * MAX_STEPS_PER_SEC

        # Accumulate fractional wheel steps; emit whole steps
        wheel_accum += steps_per_sec * dt

        # mouse.wheel expects an integer "amount"
        emit = int(wheel_accum)
        if emit != 0:
            mouse.wheel(emit)     # positive=up, negative=down
            wheel_accum -= emit

        # --- UI text ---
        cv2.putText(frame, f"Neutral: {NEUTRAL_DIST:.3f}  Deadzone: {DEADZONE:.3f}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if pinch_dist is not None:
            cv2.putText(frame, f"Pinch dist: {pinch_dist:.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Target speed: {target_speed:+.2f}  Smoothed: {smoothed_speed:+.2f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Mode: {mode}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Smooth Pinch Zoom (scroll wheel)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
