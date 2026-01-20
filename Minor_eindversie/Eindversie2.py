import cv2
import mediapipe as mp
import keyboard
import time

print("Minor hand control - use your hand(s) to trigger keyboard actions!")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Resolutie (mag je aanpassen)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# Instellingen (relatief)
# =========================
HOLD_TIME = 1.5
CONFIRM_COOLDOWN = 1.0
DECAY_PER_SEC = 1.2

BLOCK_HEIGHT_RATIO = 0.28   # % van schermhoogte
BLOCK_WIDTH_RATIO  = 0.45   # % van schermbreedte
SPACING_RATIO      = 0.06

# =========================
# Helpers
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h

def lerp(a, b, t):
    return a + (b - a) * t

def color_red_to_green(t):
    t = clamp(t, 0.0, 1.0)
    return (0, int(255 * t), int(255 * (1 - t)))

def hand_center_px(hand_landmarks, w, h):
    ids = [0, 5, 9, 13, 17]
    cx = int(sum(hand_landmarks.landmark[i].x for i in ids) / len(ids) * w)
    cy = int(sum(hand_landmarks.landmark[i].y for i in ids) / len(ids) * h)
    return cx, cy

# =========================
# Squares
# =========================
squares = []

def init_squares(w, h):
    global squares

    block_w = int(w * BLOCK_WIDTH_RATIO)
    block_h = int(h * BLOCK_HEIGHT_RATIO)
    spacing = int(h * SPACING_RATIO)

    total_h = block_h * 2 + spacing
    start_y = h // 2 - total_h // 2
    start_x = w // 2 - block_w // 2

    rects = [
        (start_x, start_y, block_w, block_h),
        (start_x, start_y + block_h + spacing, block_w, block_h),
    ]

    actions = [("Route A", "a"), ("Route B", "b")]

    squares.clear()
    for rect, (label, key) in zip(rects, actions):
        squares.append({
            "rect": rect,
            "progress": 0.0,
            "confirmed_until": 0.0,
            "label": label,
            "key": key
        })

# =========================
# Main loop
# =========================
with mp_hands.Hands(
    max_num_hands=4,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    prev_t = time.time()
    initialized = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if not initialized:
            init_squares(w, h)
            initialized = True

        now = time.time()
        dt = now - prev_t
        prev_t = now

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        centers = []
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                centers.append(hand_center_px(hl, w, h))

        build_rate = dt / HOLD_TIME
        decay_rate = dt * (DECAY_PER_SEC / HOLD_TIME)

        for sq in squares:
            rect = sq["rect"]
            inside = sum(1 for c in centers if point_in_rect(*c, rect))
            cooldown = now < sq["confirmed_until"]

            if not cooldown:
                if inside >= 2:
                    sq["progress"] += build_rate
                elif inside == 1:
                    sq["progress"] = min(sq["progress"] + build_rate, 0.5)
                else:
                    sq["progress"] -= decay_rate

                sq["progress"] = clamp(sq["progress"], 0, 1)

                if sq["progress"] >= 1.0 and inside >= 2:
                    keyboard.press_and_release(sq["key"])
                    sq["confirmed_until"] = now + CONFIRM_COOLDOWN
            else:
                sq["progress"] = 1.0

        overlay = frame.copy()

        for sq in squares:
            x, y, sw, sh = sq["rect"]
            p = sq["progress"]
            confirmed = now < sq["confirmed_until"]
            color = (0, 255, 0) if confirmed else color_red_to_green(p)

            fill_h = int(sh * p)
            cv2.rectangle(overlay, (x, y + sh - fill_h), (x + sw, y + sh), color, -1)
            cv2.rectangle(frame, (x, y), (x + sw, y + sh), color, 3)

            font_scale = sw / 600
            (tw, th), _ = cv2.getTextSize(sq["label"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.putText(frame, sq["label"],
                        (x + sw // 2 - tw // 2, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            if confirmed:
                cv2.putText(frame, "CONFIRMED",
                            (x + sw // 2 - tw, y + sh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, color, 3)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.imshow("Hand hotspots", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


# versie met 2 knoppen: Route A en Route B