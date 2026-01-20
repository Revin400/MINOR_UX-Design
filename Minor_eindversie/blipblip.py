import cv2
import mediapipe as mp
import keyboard
import time

print("Minor hand control - hotspots + gestures (3 blocks)")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =====================
# Hotspot gedrag
# =====================
HOLD_TIME = 1.5
CONFIRM_COOLDOWN = 6.0
DECAY_PER_SEC = 1.2

# =====================
# UI schaal
# =====================
BLOCK_HEIGHT_RATIO = 0.28
BLOCK_WIDTH_RATIO  = 0.38
SPACING_RATIO      = 0.06

# =====================
# Gesture gedrag
# =====================
GESTURE_HOLD = 0.25
GESTURE_COOLDOWN = 0.8

NEXT_KEY = "space"


def clamp(x, a, b):
    return max(a, min(b, x))


def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def color_red_to_green(t):
    t = clamp(t, 0.0, 1.0)
    return (0, int(255 * t), int(255 * (1 - t)))


def hand_center_px(hand_landmarks, w, h):
    ids = [0, 5, 9, 13, 17]
    cx = int(sum(hand_landmarks.landmark[i].x for i in ids) / len(ids) * w)
    cy = int(sum(hand_landmarks.landmark[i].y for i in ids) / len(ids) * h)
    return cx, cy


def count_extended_fingers(hand_landmarks, handedness_label: str) -> int:
    lm = hand_landmarks.landmark

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    extended = 0
    for t, p in zip(tips, pips):
        if lm[t].y < lm[p].y:
            extended += 1

    if handedness_label.lower() == "right":
        if lm[4].x < lm[3].x:
            extended += 1
    else:
        if lm[4].x > lm[3].x:
            extended += 1

    return extended


def gesture_from_extended_count(extended_count: int) -> str:
    if extended_count <= 1:
        return "FIST"
    if extended_count >= 4:
        return "PALM"
    return "OTHER"


# =====================
# Hotspots
# =====================
squares = []


def init_squares(w, h):
    global squares

    block_w = int(w * BLOCK_WIDTH_RATIO)
    block_h = int(h * BLOCK_HEIGHT_RATIO)
    spacing = int(h * SPACING_RATIO)

    top_y = int(h * 0.12)

    left_x  = w // 2 - block_w - spacing // 2
    right_x = w // 2 + spacing // 2

    bottom_y = top_y + block_h + spacing
    center_x = w // 2 - block_w // 2

    rects = [
        (left_x,   top_y,    block_w, block_h),   # Route A
        (right_x,  top_y,    block_w, block_h),   # Route B
        (center_x, bottom_y, block_w, block_h),   # NEXT
    ]

    actions = [
        ("Route A", "a"),
        ("Route B", "b"),
        ("NEXT", NEXT_KEY),
    ]

    squares.clear()
    for rect, (label, key) in zip(rects, actions):
        squares.append({
            "rect": rect,
            "progress": 0.0,
            "confirmed_until": 0.0,
            "label": label,
            "key": key
        })


# =====================
# Gesture state
# =====================
last_gesture = "NONE"
gesture_stable_for = 0.0
gesture_blocked_until = 0.0


with mp_hands.Hands(
    max_num_hands=2,
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

        # =====================
        # Hand centers
        # =====================
        centers = []
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                centers.append(hand_center_px(hl, w, h))

        # =====================
        # Hotspot logic
        # =====================
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

        # =====================
        # Gesture detect
        # =====================
        gesture_text = "NONE"
        if results.multi_hand_landmarks and results.multi_handedness:
            hl = results.multi_hand_landmarks[0]
            handed = results.multi_handedness[0].classification[0].label
            ext = count_extended_fingers(hl, handed)
            g = gesture_from_extended_count(ext)
            gesture_text = f"{g} ({ext} fingers)"

            if g == last_gesture:
                gesture_stable_for += dt
            else:
                last_gesture = g
                gesture_stable_for = 0.0

            if now >= gesture_blocked_until:
                if g == "FIST" and gesture_stable_for >= GESTURE_HOLD:
                    keyboard.press_and_release(NEXT_KEY)
                    gesture_blocked_until = now + GESTURE_COOLDOWN
                    gesture_stable_for = 0.0

        # =====================
        # Draw UI
        # =====================
        overlay = frame.copy()

        for sq in squares:
            x, y, sw, sh = sq["rect"]
            p = sq["progress"]
            confirmed = now < sq["confirmed_until"]
            color = (0, 255, 0) if confirmed else color_red_to_green(p)

            fill_h = int(sh * p)
            cv2.rectangle(overlay, (x, y + sh - fill_h), (x + sw, y + sh), color, -1)
            cv2.rectangle(frame, (x, y), (x + sw, y + sh), color, 3)

            font_scale = max(0.6, sw / 600)
            (tw, th), _ = cv2.getTextSize(sq["label"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.putText(frame, sq["label"],
                        (x + sw // 2 - tw // 2, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            if confirmed:
                text = "CONFIRMED"
                (tw2, th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 3)
                cv2.putText(frame, text,
                            (x + sw // 2 - tw2 // 2, y + sh // 2 + th2 // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 255, 0), 3)

        cv2.rectangle(frame, (12, 12), (540, 56), (0, 0, 0), -1)
        cv2.putText(frame, f"Gesture: {gesture_text}", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if now < gesture_blocked_until:
            cv2.putText(frame, "Gesture cooldown", (20, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.imshow("Hotspots + Gestures", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
