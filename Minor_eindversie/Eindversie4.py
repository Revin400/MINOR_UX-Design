import cv2
import mediapipe as mp
import keyboard
import time
import numpy as np
import os

print("Hand hotspots + Heatmap history (2 hands confirm). Q=quit, N=back to choice, R=reset heatmaps")

# =========================
# Camera / MediaPipe
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# Instellingen
# =========================
HOLD_TIME = 1.5
CONFIRM_COOLDOWN = 1.0
DECAY_PER_SEC = 1.2

# UI schaal
BLOCK_HEIGHT_RATIO = 0.26
BLOCK_WIDTH_RATIO  = 0.55
SPACING_RATIO      = 0.06

# Heatmap
HEATMAP_RES = (240, 240)      # (h, w) interne heatmap resolutie per optie
HEAT_SIGMA = 10.0             # hoe "blurred" de punt wordt
HEAT_ADD = 1.0                # hoeveel intensiteit per keuze
HEAT_DECAY = 0.0              # optioneel: bv 0.0005 voor langzaam vervagen per frame

SAVE_A = "heatmap_A.npy"
SAVE_B = "heatmap_B.npy"

# =========================
# Helpers
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def point_in_rect(px, py, rect):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h

def hand_center_px(hand_landmarks, w, h):
    ids = [0, 5, 9, 13, 17]  # palm-ish
    cx = int(sum(hand_landmarks.landmark[i].x for i in ids) / len(ids) * w)
    cy = int(sum(hand_landmarks.landmark[i].y for i in ids) / len(ids) * h)
    return cx, cy

def color_red_to_green(t):
    t = clamp(t, 0.0, 1.0)
    return (0, int(255 * t), int(255 * (1 - t)))  # BGR

def put_center_text(img, text, cx, cy, font_scale, color, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# =========================
# Heatmap opslag/load
# =========================
def load_heatmap(path):
    if os.path.exists(path):
        try:
            hm = np.load(path)
            if hm.shape == HEATMAP_RES:
                return hm.astype(np.float32)
        except Exception:
            pass
    return np.zeros(HEATMAP_RES, dtype=np.float32)

def save_heatmap(path, hm):
    np.save(path, hm.astype(np.float32))

heat_A = load_heatmap(SAVE_A)
heat_B = load_heatmap(SAVE_B)

def add_gaussian(heat, ux, uy, amount=1.0, sigma=10.0):
    """
    ux,uy: normalized coords (0..1) in heatmap space.
    """
    hh, ww = heat.shape
    x0 = int(clamp(ux, 0.0, 1.0) * (ww - 1))
    y0 = int(clamp(uy, 0.0, 1.0) * (hh - 1))

    # region around point
    rad = int(3 * sigma)
    x1 = max(0, x0 - rad); x2 = min(ww, x0 + rad + 1)
    y1 = max(0, y0 - rad); y2 = min(hh, y0 + rad + 1)

    xs = np.arange(x1, x2) - x0
    ys = np.arange(y1, y2) - y0
    xx, yy = np.meshgrid(xs, ys)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32)

    heat[y1:y2, x1:x2] += amount * g

def render_heatmap_block(heat, target_w, target_h):
    """
    Render heatmap to BGR image with colormap.
    """
    hm = heat.copy()
    hm = np.clip(hm, 0, None)
    maxv = float(hm.max()) if hm.size else 0.0
    if maxv > 1e-6:
        hm = (hm / maxv) * 255.0
    hm_u8 = hm.astype(np.uint8)

    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)  # nice heatmap
    colored = cv2.resize(colored, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return colored

# =========================
# Squares
# =========================
squares = []
heat_rect = None

def init_layout(w, h):
    """
    2 keuze-blokken (boven elkaar) + heatmap-blok (onder) dat zichtbaar wordt na keuze.
    """
    global squares, heat_rect

    block_w = int(w * BLOCK_WIDTH_RATIO)
    block_h = int(h * BLOCK_HEIGHT_RATIO)
    spacing = int(h * SPACING_RATIO)

    # Heatmap block is ook een "panel" (iets lager en iets kleiner)
    heat_h = int(h * 0.28)
    heat_w = block_w
    # total layout hoogte als alles zichtbaar is (2 keuzes + spacing + heatmap + spacing)
    total_h = block_h * 2 + spacing + heat_h + spacing

    start_y = h // 2 - total_h // 2
    start_x = w // 2 - block_w // 2

    rects = [
        (start_x, start_y, block_w, block_h),  # A
        (start_x, start_y + block_h + spacing, block_w, block_h),  # B
    ]

    heat_rect = (
        start_x,
        start_y + block_h * 2 + spacing * 2,
        heat_w,
        heat_h
    )

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
# State
# =========================
initialized = False
chosen_label = None   # "Route A" / "Route B"
chosen_key = None     # "a" / "b"
chosen_rect = None    # rect of chosen block

# =========================
# Main
# =========================
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
) as hands:

    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if not initialized:
            init_layout(w, h)
            initialized = True

        now = time.time()
        dt = now - prev_t
        prev_t = now

        # optional heat decay (bv voor "levende" heatmap)
        if HEAT_DECAY > 0:
            heat_A *= (1.0 - HEAT_DECAY)
            heat_B *= (1.0 - HEAT_DECAY)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        centers = []
        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                cx, cy = hand_center_px(hl, w, h)
                centers.append((cx, cy))
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)

        build_rate = dt / HOLD_TIME
        decay_rate = dt * (DECAY_PER_SEC / HOLD_TIME)

        # =========================
        # Update keuzes alleen als nog niet gekozen
        # =========================
        if chosen_label is None:
            for sq in squares:
                rect = sq["rect"]
                cooldown = now < sq["confirmed_until"]
                inside_count = sum(1 for (cx, cy) in centers if point_in_rect(cx, cy, rect))

                if not cooldown:
                    if inside_count >= 2:
                        sq["progress"] += build_rate
                        sq["progress"] = clamp(sq["progress"], 0.0, 1.0)
                    elif inside_count == 1:
                        # max 50%
                        if sq["progress"] < 0.5:
                            sq["progress"] += build_rate
                            sq["progress"] = clamp(sq["progress"], 0.0, 0.5)
                        elif sq["progress"] > 0.5:
                            sq["progress"] -= decay_rate
                            sq["progress"] = clamp(sq["progress"], 0.5, 1.0)
                    else:
                        sq["progress"] -= decay_rate
                        sq["progress"] = clamp(sq["progress"], 0.0, 1.0)

                    # Confirm alleen als 2+ handen
                    if sq["progress"] >= 1.0 and inside_count >= 2:
                        # trigger key
                        keyboard.press_and_release(sq["key"])
                        sq["confirmed_until"] = now + CONFIRM_COOLDOWN
                        sq["progress"] = 1.0

                        # set choice state
                        chosen_label = sq["label"]
                        chosen_key = sq["key"]
                        chosen_rect = rect

                        # neem een punt: gemiddelde van hand-centers die binnen zitten
                        inside_pts = [(cx, cy) for (cx, cy) in centers if point_in_rect(cx, cy, rect)]
                        if inside_pts:
                            avgx = sum(p[0] for p in inside_pts) / len(inside_pts)
                            avgy = sum(p[1] for p in inside_pts) / len(inside_pts)
                            x, y, rw, rh = rect
                            ux = (avgx - x) / rw
                            uy = (avgy - y) / rh

                            if chosen_key == "a":
                                add_gaussian(heat_A, ux, uy, amount=HEAT_ADD, sigma=HEAT_SIGMA)
                                save_heatmap(SAVE_A, heat_A)
                            else:
                                add_gaussian(heat_B, ux, uy, amount=HEAT_ADD, sigma=HEAT_SIGMA)
                                save_heatmap(SAVE_B, heat_B)
                else:
                    sq["progress"] = 1.0

        # =========================
        # Draw UI
        # =========================
        overlay = frame.copy()

        # Draw choice blocks
        for sq in squares:
            x, y, sw, sh = sq["rect"]
            p = sq["progress"]
            confirmed = now < sq["confirmed_until"]

            if chosen_label is not None:
                # na keuze: highlight gekozen, dim de rest
                if sq["label"] == chosen_label:
                    outline = (0, 255, 0)
                else:
                    outline = (80, 80, 80)
            else:
                outline = (0, 255, 0) if confirmed else color_red_to_green(p)

            # fill progress alleen als nog niet gekozen
            if chosen_label is None:
                fill_h = int(sh * p)
                cv2.rectangle(overlay, (x, y + sh - fill_h), (x + sw, y + sh), outline, -1)
            else:
                # lichte fill als gekozen
                if sq["label"] == chosen_label:
                    cv2.rectangle(overlay, (x, y), (x + sw, y + sh), (0, 255, 0), -1)

            cv2.rectangle(frame, (x, y), (x + sw, y + sh), outline, 3)

            font_scale = max(0.7, sw / 650)
            cv2.putText(frame, sq["label"], (x + 10, y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, outline, 2)

            # progress bar onder
            bar_y = y + sh + 12
            cv2.rectangle(frame, (x, bar_y), (x + sw, bar_y + 12), (50, 50, 50), 2)
            cv2.rectangle(frame, (x, bar_y), (x + int(sw * p), bar_y + 12), outline, -1)

            if chosen_label is None and confirmed:
                put_center_text(frame, "CONFIRMED", x + sw/2, y + sh/2, font_scale * 1.1, (0, 255, 0), 3)

        # Draw heatmap block (alleen na keuze)
        hx, hy, hw, hh = heat_rect
        cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (255, 255, 255), 2)

        if chosen_label is None:
            # hint text
            put_center_text(frame, "Heatmap verschijnt na een keuze", hx + hw/2, hy + hh/2, 0.8, (255, 255, 255), 2)
        else:
            hm = heat_A if chosen_key == "a" else heat_B
            hm_img = render_heatmap_block(hm, hw, hh)

            # blend heatmap in overlay
            overlay[hy:hy+hh, hx:hx+hw] = hm_img

            cv2.putText(frame, f"Heatmap: {chosen_label} (historie)", (hx + 10, hy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame, "N = terug, R = reset heatmaps", (hx + 10, hy + hh + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # blend overlay (incl progress fills + heatmap)
        frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0)

        # Top HUD
        cv2.rectangle(frame, (12, 12), (12 + 560, 12 + 42), (0, 0, 0), -1)
        status = "Kies Route A of B (2 handen om te bevestigen)" if chosen_label is None else f"Gekozen: {chosen_label}"
        cv2.putText(frame, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Hotspots + Heatmap", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("n"):
            # terug naar keuze zonder reset (heatmap blijft)
            chosen_label = None
            chosen_key = None
            chosen_rect = None
            for sq in squares:
                sq["progress"] = 0.0
                sq["confirmed_until"] = 0.0
        if k == ord("r"):
            # reset heatmaps + terug naar keuze
            heat_A[:] = 0.0
            heat_B[:] = 0.0
            save_heatmap(SAVE_A, heat_A)
            save_heatmap(SAVE_B, heat_B)
            chosen_label = None
            chosen_key = None
            chosen_rect = None
            for sq in squares:
                sq["progress"] = 0.0
                sq["confirmed_until"] = 0.0

cap.release()
cv2.destroyAllWindows()
