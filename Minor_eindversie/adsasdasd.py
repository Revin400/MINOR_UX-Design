import cv2
import mediapipe as mp
import keyboard
import time

print("Hand control - 3 blocks (Route A / Route B / NEXT) - trigger only with 2 hands inside")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =====================
# Behavior
# =====================
HOLD_TIME = 1.2          
COOLDOWN = 1.0           

NEXT_KEY = "space"

# =====================
# UI scale (same feel as your old version)
# =====================
BLOCK_HEIGHT_RATIO = 0.28
BLOCK_WIDTH_RATIO  = 0.38
SPACING_RATIO      = 0.06

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

# =====================
# Blocks
# =====================
blocks = []

def init_blocks(w, h):
    global blocks

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

    blocks = []
    for rect, (label, key) in zip(rects, actions):
        blocks.append({
            "rect": rect,
            "label": label,
            "key": key,
            "hold": 0.0,
            "blocked_until": 0.0,
        })

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
            init_blocks(w, h)
            initialized = True

        now = time.time()
        dt = now - prev_t
        prev_t = now

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # collect hand centers
        centers = []
        num_hands = 0
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                centers.append(hand_center_px(hl, w, h))

        two_hands_present = (num_hands >= 2)

        # Evaluate each block independently:
        # Only trigger if BOTH hands are in that block for HOLD_TIME, and block not on cooldown.
        for b in blocks:
            rect = b["rect"]
            cooldown = now < b["blocked_until"]

            if not two_hands_present or cooldown:
                b["hold"] = 0.0
                continue

            inside_count = sum(1 for c in centers[:2] if point_in_rect(c[0], c[1], rect))

            if inside_count >= 2:
                b["hold"] += dt
                if b["hold"] >= HOLD_TIME:
                    keyboard.press_and_release(b["key"])
                    b["blocked_until"] = now + COOLDOWN
                    b["hold"] = 0.0
            else:
                b["hold"] = 0.0

        # =====================
        # Draw UI
        # =====================
        overlay = frame.copy()

        for b in blocks:
            x, y, bw, bh = b["rect"]
            p = clamp(b["hold"] / HOLD_TIME, 0.0, 1.0)
            cooldown = now < b["blocked_until"]

            color = (0, 255, 0) if cooldown else color_red_to_green(p)

            # fill
            fill_h = int(bh * p)
            cv2.rectangle(overlay, (x, y + bh - fill_h), (x + bw, y + bh), color, -1)
            # border
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)

            # label
            font_scale = max(0.6, bw / 600)
            (tw, th), _ = cv2.getTextSize(b["label"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.putText(frame, b["label"],
                        (x + bw // 2 - tw // 2, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # confirmed text during cooldown
            if cooldown:
                text = "CONFIRMED"
                (tw2, th2), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, 3)
                cv2.putText(frame, text,
                            (x + bw // 2 - tw2 // 2, y + bh // 2 + th2 // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 255, 0), 3)

        # top-left info
        cv2.rectangle(frame, (12, 12), (680, 88), (0, 0, 0), -1)
        cv2.putText(frame, f"Hands detected: {num_hands}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if not two_hands_present:
            msg = "Waiting for 2 hands..."
        else:
            msg = "Put BOTH hands in Route A / Route B / NEXT and hold"
        cv2.putText(frame, msg, (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        cv2.imshow("3 Blocks - 2 Hands -> Keypress", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
