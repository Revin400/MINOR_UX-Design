import cv2
import mediapipe as mp
import keyboard
import time

print("Minor hand control - use your hand(s) to trigger keyboard actions!")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ===== Instellingen =====
HOLD_TIME = 1.5
CONFIRM_COOLDOWN = 1.0
DECAY_PER_SEC = 1.2

SQUARE_SIZE = 400

# ===== Helpers =====
def clamp(x, a, b):
    return max(a, min(b, x))

def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h

def lerp(a, b, t):
    return a + (b - a) * t

def color_red_to_green(t):
    t = clamp(t, 0.0, 1.0)
    r = int(lerp(255, 0, t))
    g = int(lerp(0, 255, t))
    return (0, g, r)

def hand_center_px(hand_landmarks, frame_w, frame_h):
    ids = [0, 5, 9, 13, 17]
    cx = int(sum(hand_landmarks.landmark[i].x for i in ids) / len(ids) * frame_w)
    cy = int(sum(hand_landmarks.landmark[i].y for i in ids) / len(ids) * frame_h)
    return cx, cy

# ===== Hotspots =====
squares = []

def init_squares(frame_w, frame_h):
    """
    2 blokken:
    - samen horizontaal gecentreerd
    - verticaal exact in het midden
    """
    global squares

    spacing = 80
    total_width = SQUARE_SIZE * 2 + spacing
    start_x = frame_w // 2 - total_width // 2
    center_y = frame_h // 2 - SQUARE_SIZE // 2

    rects = [
        (start_x, center_y, SQUARE_SIZE, SQUARE_SIZE),                       # Left
        (start_x + SQUARE_SIZE + spacing, center_y, SQUARE_SIZE, SQUARE_SIZE) # Right
    ]

    actions = [
        ("Route A", "a"),
        ("Route B", "b"),
    ]

    squares = []
    for rect, (label, key) in zip(rects, actions):
        squares.append({
            "rect": rect,
            "progress": 0.0,
            "confirmed_until": 0.0,
            "label": label,
            "key": key
        })

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    prev_t = time.time()
    inited = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if not inited:
            init_squares(w, h)
            inited = True

        now = time.time()
        dt = now - prev_t
        prev_t = now

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Hand centers
        centers = []
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                cx, cy = hand_center_px(hl, w, h)
                centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

        build_rate = dt / HOLD_TIME
        decay_rate = dt * (DECAY_PER_SEC / HOLD_TIME)

        # Update progress
        for sq in squares:
            rect = sq["rect"]
            inside_count = sum(1 for (cx, cy) in centers if point_in_rect(cx, cy, rect))
            in_cooldown = now < sq["confirmed_until"]

            if not in_cooldown:
                if inside_count >= 2:
                    sq["progress"] += build_rate
                elif inside_count == 1:
                    sq["progress"] = min(sq["progress"] + build_rate, 0.5)
                else:
                    sq["progress"] -= decay_rate

                sq["progress"] = clamp(sq["progress"], 0.0, 1.0)

                if sq["progress"] >= 1.0 and inside_count >= 2:
                    keyboard.press_and_release(sq["key"])
                    sq["confirmed_until"] = now + CONFIRM_COOLDOWN
            else:
                sq["progress"] = 1.0

        # Draw UI
        overlay = frame.copy()

        for sq in squares:
            x, y, sw, sh = sq["rect"]
            p = sq["progress"]
            confirmed = now < sq["confirmed_until"]
            color = (0, 255, 0) if confirmed else color_red_to_green(p)

            fill_h = int(sh * p)
            cv2.rectangle(overlay, (x, y + sh - fill_h), (x + sw, y + sh), color, -1)
            cv2.rectangle(frame, (x, y), (x + sw, y + sh), color, 3)

            cv2.putText(frame, sq["label"], (x, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if confirmed:
                cv2.putText(frame, "CONFIRMED",
                            (x + 60, y + sh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 3)

        frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)
        cv2.imshow("Hand hotspots", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
