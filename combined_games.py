import time
import math
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import sys

# --- Helpers / shared config -----------------------------------------------------
pyautogui.FAILSAFE = False # disable moving mouse to corner to stop pyautogui

# Finger landmarks
FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "little": 20} # tip,dip,pip,mcp
FINGER_PIP  = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "little": 18} # pip joint indices

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1]) #calculating distance for measuring if finger is bent 
                                            #or no w.r.t another point in same finger or wrist

def normalized_to_pixel(norm_landmark, image_w, image_h):      #converting normalized landmark coordinates to pixel coordinates
    x = min(max(int(norm_landmark.x * image_w), 0), image_w - 1)# min and max are used to ensure that the coordinates are within the image bounds
    y = min(max(int(norm_landmark.y * image_h), 0), image_h - 1)
    return x, y

def estimate_hand_size(landmarks, w, h):  #calculating hand size based on the distance between wrist and mid MCP
    wrist = normalized_to_pixel(landmarks[0], w, h) # wrist is the first landmark in the hand landmarks
    mid_mcp = normalized_to_pixel(landmarks[9], w, h) # to make independent of distance of hand from camera
    return dist(wrist, mid_mcp) + 1e-6  #codes use "normalized_value = finger_distance / hand_size" to avoid denominator to become 0 we add a small value of 1e-6

def is_finger_bent(landmarks, w, h, finger_name, hand_size):
    tip_idx = FINGER_TIPS[finger_name] # getting tip and pip index of the finger
    pip_idx = FINGER_PIP[finger_name] # pip is the joint below the tip
    tip = normalized_to_pixel(landmarks[tip_idx], w, h) # converting normalized landmarks to pixel coordinates
    pip = normalized_to_pixel(landmarks[pip_idx], w, h)
    if finger_name != "thumb": # thumb bending detection is different from other fingers
        bent = tip[1] > pip[1]
        wrist = normalized_to_pixel(landmarks[0], w, h)
        if dist(tip, wrist) < hand_size * 0.35:# distance threshold to detect bent thumb 
            bent = True
        if dist(tip, pip) < hand_size * 0.25:#lower value more accurate bending detection but shd bend more
            bent = True
        return bent
    index_mcp = normalized_to_pixel(landmarks[5], w, h)
    tip = normalized_to_pixel(landmarks[4], w, h)
    return dist(tip, index_mcp) < hand_size * 0.45 #higher value less accurate bending detection but slightly bent finger can be detected

# --- Subway Surf: single-hand implementation -------------------------------
def run_subway_single(cam_id=0):
    # single-hand mapping
    FINGER_ACTION = {"thumb": "right", "ring": "left", "index": "up", "middle": "down"}
    OVERBOARD_CLICK_COUNT = 2 #once little finger is bent it will perform double click
    OVERBOARD_CLICK_INTERVAL = 0.12 #interval between the two clicks
    GAME_CLICK_POS = None # (x,y) position to click for overboard action, None means current mouse position
    DETECTION_CONF = 0.45 #minimum confidence value for hand detection to be considered successful  ""'“Only detect a hand if MediaPipe is at least 45% sure.”'""
    TRACKING_CONF = 0.45 #minimum confidence value for hand tracking to be considered successful
    MAX_HANDS = 1 # only one hand
    DEBOUNCE_MS = 10 # debounce time in milliseconds to prevent multiple triggers from a single bend

    mp_hands = mp.solutions.hands  #MEDIAPIPE to detect hand and landmarks we use mediapipe librarys
    mp_drawing = mp.solutions.drawing_utils  #Landmarks (21 dots), Connections (lines between dots), Skeleton of the hand

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)#opens your camera (webcam) so OpenCV can read frames from it.
    if not cap.isOpened(): # checking if camera opened successfully  ###0 → Default laptop/PC webcam, 1 → External USB camera, 2 → Second external camera (if connected)
        print("Cannot open camera.") #cv2.cap should be successfully opened to proceed further “Use the DirectShow driver to access the webcam.”
        return

    prev_bent_state = {f: False for f in FINGER_TIPS.keys()}# This prevents multiple unwanted triggers
    last_trigger_time = {f: 0 for f in FINGER_TIPS.keys()}#initializing last trigger time for each finger to 0 (starting every finger initialized to 0)

    with mp_hands.Hands(static_image_mode=False,     #static_image_mode=False means it will treat the input images as a video stream.
                        max_num_hands=MAX_HANDS,     #max_num_hands=1 means it will only detect one hand in the frame.
                        min_detection_confidence=DETECTION_CONF, #minimum confidence value for hand detection to be considered successful
                        min_tracking_confidence=TRACKING_CONF) as hands:  #minimum confidence value for hand tracking to be considered successful
        while True:
            ret, frame = cap.read()
            if not ret:
                break    #if frame is not read correctly, break the loop
            frame = cv2.flip(frame, 1) #flipping the frame horizontally for mirror-like effect
            h, w = frame.shape[:2]  #getting height and width of the frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converting BGR image (OpenCV default) to RGB (Mediapipe requirement)
            results = hands.process(rgb)  #processing the RGB frame to detect hands and landmarks
            now_ms = time.time() * 1000 #current time in milliseconds
            info = [] #list to store info messages for display

            if results.multi_hand_landmarks: #if hand landmarks are detected
                hand_landmarks = results.multi_hand_landmarks[0] #get the first detected hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #drawing landmarks and connections on the frame for visualization
                landmarks = hand_landmarks.landmark #getting the list of landmarks for the detected hand
                hand_size = estimate_hand_size(landmarks, w, h) #estimating hand size for bend detection

                # label tips 1..5
                for i, name in enumerate(["thumb", "index", "middle", "ring", "little"], start=1): #labeling each finger tip with its name and index number
                    x, y = normalized_to_pixel(landmarks[FINGER_TIPS[name]], w, h) #getting pixel coordinates of the finger tip
                    cv2.putText(frame, str(i), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2) #drawing the index number near the finger tip

                bent_state = {name: is_finger_bent(landmarks, w, h, name, hand_size) for name in FINGER_TIPS} #determining bent state for each finger

                # single-key actions
                for finger, key in FINGER_ACTION.items(): #mapping finger bends to game actions
                    bent = bent_state.get(finger, False) #checking if the finger is bent
                    if bent and not prev_bent_state[finger] and (now_ms - last_trigger_time[finger] > DEBOUNCE_MS): #if finger is bent and was not bent previously and debounce time has passed
                        try: # sending key press to the game
                            pyautogui.press(key) #pressing the corresponding key for the bent finger
                        except Exception: # handling any exceptions that may occur during key press
                            pass
                        last_trigger_time[finger] = now_ms #updating last trigger time for the finger
                        info.append(f"{finger} -> {key}") #adding info message for display
                    prev_bent_state[finger] = bent #updating previous bent state for the finger

                # little -> overboard (double-click)
                little_bent = bent_state.get("little", False) #checking if little finger is bent
                if little_bent and not prev_bent_state.get("little", False) and (now_ms - last_trigger_time.get("little",0) > DEBOUNCE_MS): #if little finger is bent and was not bent previously and debounce time has passed
                    try:
                        if GAME_CLICK_POS: #if specific click position is provided
                            pyautogui.click(x=GAME_CLICK_POS[0], y=GAME_CLICK_POS[1],
                                            clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL) #performing double click at specified position
                        else:
                            pyautogui.click(clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL) #performing double click at current mouse position
                    except Exception: # handling any exceptions that may occur during click
                        pass
                    last_trigger_time["little"] = now_ms #updating last trigger time for little finger
                    info.append("little -> overboard") #adding info message for display
                prev_bent_state["little"] = little_bent #updating previous bent state for little finger

            else:
                for name in prev_bent_state:  # if no hand is detected, reset all previous bent states to False
                    prev_bent_state[name] = False # resetting previous bent state for all fingers

            # overlays
            for i, t in enumerate(info[:5]): #displaying info messages on the frame
                cv2.putText(frame, t, (10, 30 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2) # drawing each info message on the frame
            cv2.putText(frame, "Thumb=Right | Ring=Left | Index=Up | Middle=Down | Little=Overboard(double-click)", #   displaying control instructions at the bottom of the frame
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)  #drawing control instructions on the frame

            cv2.imshow("Subway Surf - single hand", frame)  #displaying the processed frame in a window
            if cv2.waitKey(1) & 0xFF == 27:   #waiting for ESC key to exit
                break

    cap.release() # releasing the camera resource
    cv2.destroyAllWindows()  # closing all OpenCV windows

# --- Subway Surf: two-hand implementation -----------------------------------
def run_subway_two(cam_id=0):  # implementing two-hand control for Subway Surf game
    # two-hand mapping
    HAND_ACTIONS = {
        "Right": {"index": "up",   "thumb": "right"},
        "Left":  {"index": "down", "thumb": "left"}
    } #mapping finger bends to game actions for each hand
    FINGERS_USED = ["index", "thumb"]
    OVERBOARD_CLICK_COUNT = 2
    OVERBOARD_CLICK_INTERVAL = 0.12
    GAME_CLICK_POS = None
    DETECTION_CONF = 0.45
    TRACKING_CONF = 0.45
    MAX_HANDS = 2
    DEBOUNCE_MS = 10

    mp_hands = mp.solutions.hands # initializing mediapipe hands solution
    mp_drawing = mp.solutions.drawing_utils # initializing mediapipe drawing utils

    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW) # opening camera for video capture
    if not cap.isOpened(): # checking if camera opened successfully
        print("Cannot open camera.") #  print error message if camera cannot be opened
        return

    prev_bent_state = {} # to track previous bent state of fingers for both hands
    last_trigger_time = {} # to track last trigger time for debounce mechanism

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        min_detection_confidence=DETECTION_CONF,
                        min_tracking_confidence=TRACKING_CONF) as hands: # initializing mediapipe hands with specified parameters
        while True:
            ret, frame = cap.read() # reading frame from camera
            if not ret: # if frame is not read correctly, break the loop
                break
            frame = cv2.flip(frame, 1) # flipping the frame horizontally for mirror-like effect
            h, w = frame.shape[:2] #    getting height and width of the frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converting BGR image (OpenCV default) to RGB (Mediapipe requirement)
            results = hands.process(rgb) # processing the RGB frame to detect hands and landmarks
            now_ms = time.time() * 1000 # current time in milliseconds
            info = [] # list to store info messages for display

            if results.multi_hand_landmarks and results.multi_handedness: # if hand landmarks are detected
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness): # iterating over detected hands and their handedness
                    hand_label = hand_handedness.classification[0].label  # "Left" or "Right"
                    landmarks = hand_landmarks.landmark # getting the list of landmarks for the detected hand
                    hand_size = estimate_hand_size(landmarks, w, h) # estimating hand size for bend detection
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # drawing landmarks and connections on the frame for visualization

                    # draw labels
                    for fname in FINGERS_USED + ["little"]: # labeling each finger tip with its name
                        tip_idx = FINGER_TIPS[fname] # getting tip index of the finger
                        x, y = normalized_to_pixel(landmarks[tip_idx], w, h) #querying pixel coordinates of the finger tip
                        label = f"{hand_label[0]}_{fname[:3]}" # creating label for the finger tip
                        cv2.putText(frame, label, (x-30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2) # drawing the label near the finger tip

                    # index & thumb actions
                    for fname in FINGERS_USED: # iterating over fingers used for actions
                        key = f"{hand_label}_{fname}" # creating unique key for each finger of each hand
                        bent = is_finger_bent(landmarks, w, h, fname, hand_size) # checking if the finger is bent
                        if bent and not prev_bent_state.get(key, False) and (now_ms - last_trigger_time.get(key,0) > DEBOUNCE_MS): #    
                            action_key = HAND_ACTIONS.get(hand_label, {}).get(fname) #  getting the corresponding action key for the bent finger
                            if action_key: #    if action key is found for the bent finger
                                try:
                                    pyautogui.press(action_key) #   pressing the corresponding key for the bent finger
                                except Exception: # handling any exceptions that may occur during key press
                                    pass
                                last_trigger_time[key] = now_ms # updating last trigger time for the finger
                                info.append(f"{key} -> {action_key}")  # adding info message for display
                        prev_bent_state[key] = bent # updating previous bent state for the finger

                    # little -> overboard (double-click)
                    little_key = f"{hand_label}_little"  # creating unique key for little finger of each hand
                    little_bent = is_finger_bent(landmarks, w, h, "little", hand_size)  #   checking if little finger is bent
                    if little_bent and not prev_bent_state.get(little_key, False) and (now_ms - last_trigger_time.get(little_key,0) > DEBOUNCE_MS): # if little finger is bent and was not bent previously and debounce time has passed
                        try:
                            if GAME_CLICK_POS: # if specific click position is provided
                                pyautogui.click(x=GAME_CLICK_POS[0], y=GAME_CLICK_POS[1], 
                                                clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL) # performing double click at specified position
                            else:
                                pyautogui.click(clicks=OVERBOARD_CLICK_COUNT, interval=OVERBOARD_CLICK_INTERVAL) # performing double click at current mouse position
                        except Exception:
                            pass
                        last_trigger_time[little_key] = now_ms  #   updating last trigger time for little finger
                        info.append(f"{little_key} -> overboard") # adding info message for display
                    prev_bent_state[little_key] = little_bent # updating previous bent state for little finger

            else:
                prev_bent_state.clear() # if no hand is detected, reset all previous bent states

            for i, t in enumerate(info[:6]): # displaying info messages on the frame
                cv2.putText(frame, t, (10, 30 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2) # drawing each info message on the frame

            cv2.putText(frame, "Right: thumb->left, index->down | Left: thumb->right, index->up | little->overboard(double-click)",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1) # drawing control instructions on the frame

            cv2.imshow("Subway Surf - two hands", frame) # displaying the processed frame in a window
            if cv2.waitKey(1) & 0xFF == 27: #   waiting for ESC key to exit
                break

    cap.release()  # releasing the camera resource
    cv2.destroyAllWindows()   # closing all OpenCV windows
# --- Hill Climb Racing implementation ----------------------------------------

# --- CLI / Menu -------------------------------------------------------------
def choose(prompt, options): # simple CLI menu
    print(prompt)  #printing the prompt message
    for k, v in options.items():  # iterating over the options dictionary
        print(f" {k}. {v}")  # printing each option with its key
    choice = input("Enter choice: ").strip()   # getting user input and stripping any leading/trailing whitespace
    return choice  # returning the user's choice


# --- Utility functions ------------------------------------------------------
def dist(p1, p2):  #    calculating Euclidean distance between two points
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])  # hypotenuse function to calculate distance

def normalized_to_pixel(lm, w, h):  # converting normalized landmark coordinates to pixel coordinates
    return int(lm.x * w), int(lm.y * h) # calculating pixel coordinates based on image width and height


# --- Hill Climb Racing Controller ------------------------------------------
def run_hill_climb(cam_id=0):  # implementing hand gesture control for Hill Climb Racing game
    CAM_ID = cam_id
    DETECTION_CONF = 0.5 #  minimum confidence value for hand detection to be considered successful
    TRACKING_CONF = 0.5
    MAX_HANDS = 1   # only one hand
    INDEX_BEND_THRESHOLD = 0.6  # higher threshold → easier to trigger
    RING_BEND_THRESHOLD = 0.6 # higher threshold → easier to trigger

    mp_hands = mp.solutions.hands  # initializing mediapipe hands solution
    mp_drawing = mp.solutions.drawing_utils   # initializing mediapipe drawing utils

    cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW) # opening camera for video capture
    if not cap.isOpened(): # checking if camera opened successfully
        print("Cannot open camera.") # print error message if camera cannot be opened
        return


    pressed = {"Index": False, "Ring": False} # to track pressed state of index and ring finger actions

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        min_detection_confidence=DETECTION_CONF,
                        min_tracking_confidence=TRACKING_CONF) as hands: # initializing mediapipe hands with specified parameters  
        while True:
            ok, frame = cap.read() # reading frame from camera
            if not ok: # if frame is not read correctly,
                break

            h, w = frame.shape[:2]  # getting height and width of the frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # converting BGR image (OpenCV default) to RGB (Mediapipe requirement)
            results = hands.process(rgb)  # processing the RGB frame to detect hands and landmarks

            if results.multi_hand_landmarks:  # if hand landmarks are detected
                for hand_landmarks in results.multi_hand_landmarks:  # iterating over detected hands
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # drawing landmarks and connections on the frame for visualization
                    landmarks = hand_landmarks.landmark  # getting the list of landmarks for the detected hand

                    # --- Index bend detection ---
                    index_tip = normalized_to_pixel(landmarks[8], w, h) # index finger tip landmark
                    index_mcp = normalized_to_pixel(landmarks[5], w, h) # index finger MCP landmark
                    index_dist = dist(index_tip, index_mcp) # calculating distance between index tip and MCP
                    hand_size = dist(normalized_to_pixel(landmarks[0], w, h),
                                     normalized_to_pixel(landmarks[9], w, h)) + 1e-6  # estimating hand size based on wrist and mid MCP distance
                    index_bent = index_dist < hand_size * INDEX_BEND_THRESHOLD # determining if index finger is bent based on distance threshold

                    # --- Ring bend detection ---
                    ring_tip = normalized_to_pixel(landmarks[16], w, h) # ring finger tip landmark
                    ring_mcp = normalized_to_pixel(landmarks[13], w, h) # ring finger MCP landmark
                    ring_dist = dist(ring_tip, ring_mcp)
                    ring_bent = ring_dist < hand_size * RING_BEND_THRESHOLD # determining if ring finger is bent based on distance threshold


                    # --- Index action → Right Arrow (acceleration) ---
                    if index_bent and not pressed["Index"]: # if index finger is bent and action is not already pressed
                        try: pyautogui.keyDown("right") #   pressing down the right arrow key for acceleration
                        except Exception: pass
                        pressed["Index"] = True # updating pressed state for index action
                    elif (not index_bent) and pressed["Index"]: #   if index finger is not bent and action is currently pressed
                        try: pyautogui.keyUp("right") # releasing the right arrow key
                        except Exception: pass
                        pressed["Index"] = False # updating pressed state for index action

                    # --- Ring action → Left Arrow (brake) ---
                    if ring_bent and not pressed["Ring"]:  #    if ring finger is bent and action is not already pressed
                        try: pyautogui.keyDown("left") # pressing down the left arrow key for brake
                        except Exception: pass
                        pressed["Ring"] = True # updating pressed state for ring action
                    elif (not ring_bent) and pressed["Ring"]: # if ring finger is not bent and action is currently pressed
                        try: pyautogui.keyUp("left") # releasing the left arrow key
                        except Exception: pass  
                        pressed["Ring"] = False # updating pressed state for ring action

                    # --- Visual feedback ---
                    cv2.putText(frame, f"Index:{'BENT' if index_bent else 'OPEN'}",
                                (index_tip[0]-40, index_tip[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0,255,0) if index_bent else (0,200,200), 2)  # drawing index finger bend status on the frame 
                    cv2.putText(frame, f"Ring:{'BENT' if ring_bent else 'OPEN'}",
                                (ring_tip[0]-40, ring_tip[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255,0,0) if ring_bent else (200,200,0), 2) # drawing ring finger bend status on the frame

            # --- Display ---
            disp = cv2.flip(frame, 1)  # flipping the frame horizontally for mirror-like effect
            cv2.putText(disp, "Index=RIGHT (Accel) | Ring=LEFT (Brake). Press ESC to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2) # drawing control instructions on the frame
            cv2.imshow("Hill Climb Racing - One Hand Control", disp)  # displaying the processed frame in a window

            if cv2.waitKey(1) & 0xFF == 27:  # waiting for ESC key to exit
                break

    # --- Release keys on exit ---
    try:
        if pressed["Index"]: pyautogui.keyUp("right")  # releasing right arrow key if it was pressed  
        if pressed["Ring"]: pyautogui.keyUp("left")   # releasing left arrow key if it was pressed
    except Exception:
        pass

    cap.release()  # releasing the camera resource
    cv2.destroyAllWindows()  # closing all OpenCV windows


def main_menu(): # main menu for selecting game mode
    while True:
        choice = choose("Select game:", {"1": "Run Subway Surf", "2": "Run Hill Climb Racing", "q": "Quit"})
        if choice == "1":
            sub_choice = choose("Subway Surf mode:", {"1": "Using 1 hand", "2": "Using 2 hands", "b": "Back"})  # submenu for selecting Subway Surf mode
            if sub_choice == "1":
                print("Starting Subway Surf (1 hand). Focus game window for key input. ESC closes.")
                run_subway_single()
            elif sub_choice == "2":
                print("Starting Subway Surf (2 hands). Focus game window for key input. ESC closes.")
                run_subway_two()
            elif sub_choice == "b":  #  going back to main menu
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

if __name__ == "__main__":  # main entry point of the program
    try:  # running the main menu
        main_menu() #   calling the main menu function
    except KeyboardInterrupt:  # handling keyboard interrupt (Ctrl+C)
        print("\nInterrupted. Exiting.")  # printing exit message
        try: #  closing OpenCV windows on exit
            cv2.destroyAllWindows() # closing all OpenCV windows
        except Exception: # handling any exceptions that may occur during window closing
            pass # exiting the program
        sys.exit(0) # exiting the program with status code 0