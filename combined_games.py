import time
import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import sys

# --- Helpers / shared config -------------------------------------------------
pyautogui.FAILSAFE = False

# Finger landmarks
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20}
FINGER_PIP  = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "little": 18}

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def normalized_to_pixel(norm_landmark, image_w, image_h):
    x = min(max(int(norm_landmark.x * image_w), 0), image_w - 1)
    y = min(max(int(norm_landmark.y * image_h), 0), image_h - 1)
    return x, y

def estimate_hand_size(landmarks, w, h):
    wrist = normalized_to_pixel(landmarks[0], w, h)
    mid_mcp = normalized_to_pixel(landmarks[9], w, h)
    return dist(wrist, mid_mcp) + 1e-6

def is_finger_bent(landmarks, w, h, finger_name, hand_size):
    tip_idx = FINGER_TIPS[finger_name]
    pip_idx = FINGER_PIP[finger_name]
    tip = normalized_to_pixel(landmarks[tip_idx], w, h)
    pip = normalized_to_pixel(landmarks[pip_idx], w, h)
    if finger_name != "thumb":
        bent = tip[1] > pip[1]
        wrist = normalized_to_pixel(landmarks[0], w, h)
        if dist(tip, wrist) < hand_size * 0.35:
            bent = True
        if dist(tip, pip) < hand_size * 0.25:
            bent = True
        return bent
    index_mcp = normalized_to_pixel(landmarks[5], w, h)
    tip = normalized_to_pixel(landmarks[4], w, h)
    return dist(tip, index_mcp) < hand_size * 0.45

# --- Subway Surf: single-hand implementation -------------------------------
def run_subway_single(cam_id=0):
    # single-hand mapping
    FINGER_ACTION = {"thumb": "right", "ring": "left", "index": "up", "middle": "down"}
    OVERBOARD_CLICK_COUNT = 2
    OVERBOARD_CLICK_INTERVAL = 0.12
    GAME_CLICK_POS = None
    DETECTION_CONF = 0.45
    TRACKING_CONF = 0.45
    MAX_HANDS = 1
    DEBOUNCE_MS = 10

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    prev_bent_state = {f: False for f in FINGER_TIPS.keys()}
    last_trigger_time = {f: 0 for f in FINGER_TIPS.keys()}

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        min_detection_confidence=DETECTION_CONF,
                        min_tracking_confidence=TRACKING_CONF) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now_ms = time.time() * 1000
            info = []

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                hand_size = estimate_hand_size(landmarks, w, h)

                # label tips 1..5
                for i, name in enumerate(["thumb", "index", "middle", "ring", "little"], start=1):
                    x, y = normalized_to_pixel(landmarks[FINGER_TIPS[name]], w, h)
                    cv2.putText(frame, str(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                bent_state = {name: is_finger_bent(landmarks, w, h, name, hand_size) for name in FINGER_TIPS}

                # single-key actions
                for finger, key in FINGER_ACTION.items():
                    bent = bent_state.get(finger, False)
                    if bent and not prev_bent_state[finger] and (now_ms - last_trigger_time[finger] > DEBOUNCE_MS):
                        try:
                            pyautogui.press(key)
                        except Exception:
                            pass
                        last_trigger_time[finger] = now_ms
                        info.append(f"{finger} -> {key}")
                    prev_bent_state[finger] = bent

                # little -> overboard (double-click)
                little_bent = bent_state.get("little", False)
                if little_bent and not prev_bent_state.get("little", False) and (now_ms - last_trigger_time.get("little",0) > DEBOUNCE_MS):
                    try:
                        if GAME_CLICK_POS:
                            pyautogui.click(x=GAME_CLICK_POS[0], y=GAME_CLICK_POS[1],
                                            clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL)
                        else:
                            pyautogui.click(clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL)
                    except Exception:
                        pass
                    last_trigger_time["little"] = now_ms
                    info.append("little -> overboard")
                prev_bent_state["little"] = little_bent

            else:
                for name in prev_bent_state:
                    prev_bent_state[name] = False

            # overlays
            for i, t in enumerate(info[:5]):
                cv2.putText(frame, t, (10, 30 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, "Thumb=Right | Ring=Left | Index=Up | Middle=Down | Little=Overboard(double-click)",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Subway Surf - single hand", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# --- Subway Surf: two-hand implementation -----------------------------------
def run_subway_two(cam_id=0):
    # two-hand mapping
    HAND_ACTIONS = {
        "Right": {"index": "up",   "thumb": "right"},
        "Left":  {"index": "down", "thumb": "left"}
    }
    FINGERS_USED = ["index", "thumb"]
    OVERBOARD_CLICK_COUNT = 2
    OVERBOARD_CLICK_INTERVAL = 0.12
    GAME_CLICK_POS = None
    DETECTION_CONF = 0.45
    TRACKING_CONF = 0.45
    MAX_HANDS = 2
    DEBOUNCE_MS = 10

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    prev_bent_state = {}
    last_trigger_time = {}

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        min_detection_confidence=DETECTION_CONF,
                        min_tracking_confidence=TRACKING_CONF) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            now_ms = time.time() * 1000
            info = []

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = hand_handedness.classification[0].label  # "Left" or "Right"
                    landmarks = hand_landmarks.landmark
                    hand_size = estimate_hand_size(landmarks, w, h)
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # draw labels
                    for fname in FINGERS_USED + ["little"]:
                        tip_idx = FINGER_TIPS[fname]
                        x, y = normalized_to_pixel(landmarks[tip_idx], w, h)
                        label = f"{hand_label[0]}_{fname[:3]}"
                        cv2.putText(frame, label, (x-30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # index & thumb actions
                    for fname in FINGERS_USED:
                        key = f"{hand_label}_{fname}"
                        bent = is_finger_bent(landmarks, w, h, fname, hand_size)
                        if bent and not prev_bent_state.get(key, False) and (now_ms - last_trigger_time.get(key,0) > DEBOUNCE_MS):
                            action_key = HAND_ACTIONS.get(hand_label, {}).get(fname)
                            if action_key:
                                try:
                                    pyautogui.press(action_key)
                                except Exception:
                                    pass
                                last_trigger_time[key] = now_ms
                                info.append(f"{key} -> {action_key}")
                        prev_bent_state[key] = bent

                    # little -> overboard (double-click)
                    little_key = f"{hand_label}_little"
                    little_bent = is_finger_bent(landmarks, w, h, "little", hand_size)
                    if little_bent and not prev_bent_state.get(little_key, False) and (now_ms - last_trigger_time.get(little_key,0) > DEBOUNCE_MS):
                        try:
                            if GAME_CLICK_POS:
                                pyautogui.click(x=GAME_CLICK_POS[0], y=GAME_CLICK_POS[1],
                                                clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL)
                            else:
                                pyautogui.click(clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL)
                        except Exception:
                            pass
                        last_trigger_time[little_key] = now_ms
                        info.append(f"{little_key} -> overboard")
                    prev_bent_state[little_key] = little_bent

            else:
                prev_bent_state.clear()

            for i, t in enumerate(info[:6]):
                cv2.putText(frame, t, (10, 30 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.putText(frame, "Right: thumb->left, index->down | Left: thumb->right, index->up | little->overboard(double-click)",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Subway Surf - two hands", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
# --- Hill Climb Racing implementation ----------------------------------------

# --- CLI / Menu -------------------------------------------------------------
def choose(prompt, options):
    print(prompt)
    for k, v in options.items():
        print(f" {k}. {v}")
    choice = input("Enter choice: ").strip()
    return choice


# --- Utility functions ------------------------------------------------------
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def normalized_to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

# --- Hill Climb Racing Controller ------------------------------------------
def run_hill_climb(cam_id=0):
    CAM_ID = cam_id
    DETECTION_CONF = 0.5
    TRACKING_CONF = 0.5
    MAX_HANDS = 1   # only one hand
    INDEX_BEND_THRESHOLD = 0.6  # higher threshold → easier to trigger
    RING_BEND_THRESHOLD = 0.6

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    pressed = {"Index": False, "Ring": False}

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        min_detection_confidence=DETECTION_CONF,
                        min_tracking_confidence=TRACKING_CONF) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark

                    # --- Index bend detection ---
                    index_tip = normalized_to_pixel(landmarks[8], w, h)
                    index_mcp = normalized_to_pixel(landmarks[5], w, h)
                    index_dist = dist(index_tip, index_mcp)
                    hand_size = dist(normalized_to_pixel(landmarks[0], w, h),
                                     normalized_to_pixel(landmarks[9], w, h)) + 1e-6
                    index_bent = index_dist < hand_size * INDEX_BEND_THRESHOLD

                    # --- Ring bend detection ---
                    ring_tip = normalized_to_pixel(landmarks[16], w, h)
                    ring_mcp = normalized_to_pixel(landmarks[13], w, h)
                    ring_dist = dist(ring_tip, ring_mcp)
                    ring_bent = ring_dist < hand_size * RING_BEND_THRESHOLD

                    # --- Index action → Right Arrow (acceleration) ---
                    if index_bent and not pressed["Index"]:
                        try: pyautogui.keyDown("right")
                        except Exception: pass
                        pressed["Index"] = True
                    elif (not index_bent) and pressed["Index"]:
                        try: pyautogui.keyUp("right")
                        except Exception: pass
                        pressed["Index"] = False

                    # --- Ring action → Left Arrow (brake) ---
                    if ring_bent and not pressed["Ring"]:
                        try: pyautogui.keyDown("left")
                        except Exception: pass
                        pressed["Ring"] = True
                    elif (not ring_bent) and pressed["Ring"]:
                        try: pyautogui.keyUp("left")
                        except Exception: pass
                        pressed["Ring"] = False

                    # --- Visual feedback ---
                    cv2.putText(frame, f"Index:{'BENT' if index_bent else 'OPEN'}",
                                (index_tip[0]-40, index_tip[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0,255,0) if index_bent else (0,200,200), 2)
                    cv2.putText(frame, f"Ring:{'BENT' if ring_bent else 'OPEN'}",
                                (ring_tip[0]-40, ring_tip[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255,0,0) if ring_bent else (200,200,0), 2)

            # --- Display ---
            disp = cv2.flip(frame, 1)
            cv2.putText(disp, "Index=RIGHT (Accel) | Ring=LEFT (Brake). Press ESC to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Hill Climb Racing - One Hand Control", disp)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    # --- Release keys on exit ---
    try:
        if pressed["Index"]: pyautogui.keyUp("right")
        if pressed["Ring"]: pyautogui.keyUp("left")
    except Exception:
        pass

    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    while True:
        choice = choose("Select game:", {"1": "Run Subway Surf", "2": "Run Hill Climb Racing", "q": "Quit"})
        if choice == "1":
            sub_choice = choose("Subway Surf mode:", {"1": "Using 1 hand", "2": "Using 2 hands", "b": "Back"})
            if sub_choice == "1":
                print("Starting Subway Surf (1 hand). Focus game window for key input. ESC closes.")
                run_subway_single()
            elif sub_choice == "2":
                print("Starting Subway Surf (2 hands). Focus game window for key input. ESC closes.")
                run_subway_two()
            elif sub_choice == "b":
                continue
            else:
                print("Invalid selection.")
        elif choice == "2":
            print("Starting Hill Climb Racing. Focus game window for key input. ESC closes.")
            run_hill_climb()
        elif choice.lower() == "q":
            print("Exit.")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)