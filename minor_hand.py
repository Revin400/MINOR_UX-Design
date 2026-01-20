import cv2
import mediapipe as mp
import keyboard
import time

print("Minor hand control - use your hand to trigger keyboard actions!")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ===== Instellingen =====
HOLD_TIME = 1.5          # seconden hand-in-vak nodig voor confirm
CONFIRM_COOLDOWN = 1.0   # seconds na confirm waarin hij niet opnieuw triggert
DECAY_PER_SEC = 1.2      # hoe snel progress terugvalt als hand eruit gaat (1.0 = even snel als opbouw)

SQUARE_SIZE = 160        # pixels
MARGIN = 60              # afstand van rand

# ===== Helpers =====
def clamp(x, a, b):
    return max(a, min(b, x))

def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return (x <= px <= x + w) and (y <= py <= y + h)

def lerp(a, b, t):
    return a + (b - a) * t

def color_red_to_green(t):
    # t: 0..1 => rood -> groen (BGR)
    t = clamp(t, 0.0, 1.0)
    r = int(lerp(255, 0, t))
    g = int(lerp(0, 255, t))
    return (0, g, r)

def hand_center_px(hand_landmarks, frame_w, frame_h):
    # Palm center: gemiddelde van wrist (0) + MCP’s (5,9,13,17)
    ids = [0, 5, 9, 13, 17]
    xs = [hand_landmarks.landmark[i].x for i in ids]
    ys = [hand_landmarks.landmark[i].y for i in ids]
    cx = int(sum(xs) / len(xs) * frame_w)
    cy = int(sum(ys) / len(ys) * frame_h)
    return cx, cy

# Per square: rect, progress, confirmed_until, label, key
squares = []

def init_squares(frame_w, frame_h):
    global squares
    # 4 vaste vakken: linksboven, rechtsboven, linksonder, rechtsonder
    rects = [
        (MARGIN, MARGIN, SQUARE_SIZE, SQUARE_SIZE),
        (frame_w - MARGIN - SQUARE_SIZE, MARGIN, SQUARE_SIZE, SQUARE_SIZE),
        (MARGIN, frame_h - MARGIN - SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
        (frame_w - MARGIN - SQUARE_SIZE, frame_h - MARGIN - SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
    ]

    # Koppel acties/toetsen aan de vier vakken
    actions = [
        ("Option A", "a"),
        ("Option B", "b"),
        ("Left", "left"),
        ("Right", "right"),
    ]

    squares = []
    for rect, (label, key) in zip(rects, actions):
        squares.append({
            "rect": rect,
            "progress": 0.0,         # 0..1
            "confirmed_until": 0.0,  # timestamp
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Verzamel hand-centers (in pixels)
        centers = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cx, cy = hand_center_px(hand_landmarks, w, h)
                centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

        # Update squares progress
        for sq in squares:
            x, y, sw, sh = sq["rect"]
            inside = any(point_in_rect(cx, cy, sq["rect"]) for (cx, cy) in centers)

            # cooldown check
            in_cooldown = now < sq["confirmed_until"]

            if inside and not in_cooldown:
                sq["progress"] += dt / HOLD_TIME
            else:
                # decay als je eruit gaat (of tijdens cooldown laten we hem visueel "confirmed")
                if not in_cooldown:
                    sq["progress"] -= dt * (DECAY_PER_SEC / HOLD_TIME)

            sq["progress"] = clamp(sq["progress"], 0.0, 1.0)

            # Confirm moment
            if sq["progress"] >= 1.0 and not in_cooldown:
                keyboard.press_and_release(sq["key"])
                sq["confirmed_until"] = now + CONFIRM_COOLDOWN
                # progress mag blijven staan op 1 tijdens confirmed
                sq["progress"] = 1.0

        # Draw squares UI
        overlay = frame.copy()

        for sq in squares:
            x, y, sw, sh = sq["rect"]
            p = sq["progress"]
            confirmed = now < sq["confirmed_until"]

            if confirmed:
                outline = (0, 255, 0)
            else:
                outline = color_red_to_green(p)

            # semi-transparante fill die meegroeit
            fill_h = int(sh * p)
            cv2.rectangle(overlay, (x, y + (sh - fill_h)), (x + sw, y + sh), outline, -1)

            # outline + label
            cv2.rectangle(frame, (x, y), (x + sw, y + sh), outline, 3)

            text = sq["label"]
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, outline, 2)

            # progress bar onderin
            bar_y = y + sh + 12
            cv2.rectangle(frame, (x, bar_y), (x + sw, bar_y + 14), (80, 80, 80), 2)
            cv2.rectangle(frame, (x, bar_y), (x + int(sw * p), bar_y + 14), outline, -1)

            if confirmed:
                cv2.putText(frame, "CONFIRMED", (x, y + sh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        # blend overlay fill
        frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

        cv2.imshow("Hand hotspots", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
