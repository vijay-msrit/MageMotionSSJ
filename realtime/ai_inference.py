import os
import cv2
import time
import numpy as np
from collections import deque

from tensorflow.keras.models import load_model

from mediapipe_utils.hand_tracker import HandTracker
from realtime.gamepad_controller import GamepadController
from realtime.smoothing import EMAFilter

# =========================================
# PROJECT ROOT
# =========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# =========================================
# LOAD MODEL
# =========================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "gesture_driver_gru.keras"
)

print("\nLoading AI model...\n")

model = load_model(MODEL_PATH)

print("AI model loaded successfully")

# =========================================
# COMPONENTS
# =========================================

tracker = HandTracker()

controller = GamepadController()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================================
# FILTERS
# =========================================

manual_steering_filter = EMAFilter(alpha=0.25)
manual_throttle_filter = EMAFilter(alpha=0.2)
manual_brake_filter = EMAFilter(alpha=0.25)

ai_steering_filter = EMAFilter(alpha=0.15)
ai_throttle_filter = EMAFilter(alpha=0.15)
ai_brake_filter = EMAFilter(alpha=0.15)

# =========================================
# SETTINGS
# =========================================

center_x = 0.28

SEQUENCE_LENGTH = 30

# -----------------------------------------
# AI BLENDING
# -----------------------------------------

STEERING_BLEND = 0.5
THROTTLE_BLEND = 0.15
BRAKE_BLEND = 0.1

# =========================================
# ADAPTIVE STEERING STABILIZER
# =========================================

previous_steering = 0.0

MAX_STEERING_DELTA = 0.18

STABILIZER_STRENGTH = 0.55

# =========================================
# MODE
# =========================================

# SEQUENCE BUFFER
# =========================================

# =========================================
# PERFORMANCE METRICS
# =========================================

prev_frame_time = 0.0
inference_latency_ms = 0.0

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)


sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)

# =========================================
# FIST DETECTION
# =========================================

def is_fist(hand):

    fingertip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]

    curled_fingers = 0

    for tip, mcp in zip(fingertip_ids, mcp_ids):

        fingertip_y = hand[tip][1]
        mcp_y = hand[mcp][1]

        if fingertip_y > mcp_y:
            curled_fingers += 1

    return curled_fingers >= 3

# =========================================
# MAIN LOOP
# =========================================


while True:

    frame_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:

    processed_frame, hands_data = tracker.process_frame(frame)

    # =====================================
    # DEFAULT VALUES
    # =====================================

    steering = 0.0
    throttle = 0.0
    brake = 0.0

    left_normalized = np.zeros(63)
    right_normalized = np.zeros(63)

    # =====================================
    # LEFT HAND -> STEERING
    # =====================================

    if "Left" in hands_data:

        left_hand_raw = np.array(
            hands_data["Left"]["raw"]
        ).reshape(21, 3)

        left_normalized = hands_data["Left"]["normalized"]

        left_fist = is_fist(left_hand_raw)

        if not left_fist:

            palm_x = np.mean([
                left_hand_raw[0][0],
                left_hand_raw[5][0],
                left_hand_raw[9][0],
                left_hand_raw[13][0],
                left_hand_raw[17][0]
            ])

            steering = (palm_x - center_x) * 5

            steering = np.clip(steering, -1, 1)

            if abs(steering) < 0.08:
                steering = 0.0

            steering = manual_steering_filter.update(
                steering
            )

    # =====================================
    # RIGHT HAND -> THROTTLE / BRAKE
    # =====================================

    if "Right" in hands_data:

        right_hand_raw = np.array(
            hands_data["Right"]["raw"]
        ).reshape(21, 3)

        right_normalized = hands_data["Right"]["normalized"]

        right_fist = is_fist(right_hand_raw)

        # ---------------------------------
        # BRAKE
        # ---------------------------------

        if right_fist:

            throttle = 0.0
            brake = 1.0

            brake = manual_brake_filter.update(
                brake
            )

        # ---------------------------------
        # THROTTLE
        # ---------------------------------

        else:

            palm_y = np.mean([
                right_hand_raw[0][1],
                right_hand_raw[5][1],
                right_hand_raw[9][1],
                right_hand_raw[13][1],
                right_hand_raw[17][1]
            ])

            throttle = np.interp(
                palm_y,
                [0.35, 0.65],
                [1.0, 0.0]
            )

            throttle = np.clip(throttle, 0, 1)

            throttle = manual_throttle_filter.update(
                throttle
            )

    # =====================================
    # BUILD FEATURE VECTOR
    # =====================================

    features = np.concatenate([
        left_normalized,
        right_normalized
    ])

    sequence_buffer.append(features)

    # =====================================
    # AI PREDICTIONS
    # =====================================

    ai_steering = steering
    ai_throttle = throttle
    ai_brake = brake


    if len(sequence_buffer) == SEQUENCE_LENGTH:

        infer_start = time.perf_counter()

        sequence = np.array(
            sequence_buffer,
            dtype=np.float32

        sequence = np.expand_dims(
            sequence,
            axis=0
        )

        prediction = model.predict(
            sequence,
            verbose=0
        prediction = model.predict(
            sequence,
            verbose=0
        )[0]

        inference_latency_ms = (time.perf_counter() - infer_start) * 1000

        ai_steering = float(prediction[0])

        # Clamp predictions
        # ---------------------------------

        ai_steering = np.clip(
            ai_steering,
            -1,
            1
        )

        ai_throttle = np.clip(
            ai_throttle,
            0,
            1
        )

        ai_brake = np.clip(
            ai_brake,
            0,
            1
        )

        # ---------------------------------
        # Smooth AI outputs
        # ---------------------------------

        ai_steering = ai_steering_filter.update(
            ai_steering
        )

        ai_throttle = ai_throttle_filter.update(
            ai_throttle
        )

        ai_brake = ai_brake_filter.update(
            ai_brake
        )

    # =====================================
    # HYBRID AI ASSIST
    # =====================================

    final_steering = steering
    final_throttle = throttle
    final_brake = brake

    if assist_mode:

        # Stronger AI steering help
        final_steering = (
            (1 - STEERING_BLEND) * steering
            + STEERING_BLEND * ai_steering
        )

        # Smaller throttle AI effect
        final_throttle = (
            (1 - THROTTLE_BLEND) * throttle
            + THROTTLE_BLEND * ai_throttle
        )

        # Brake mostly manual
        final_brake = (
            (1 - BRAKE_BLEND) * brake
            + BRAKE_BLEND * ai_brake
        )

    # =====================================
    # ADAPTIVE STEERING STABILIZER
    # =====================================

    stabilizer_active = False

    steering_delta = abs(
        final_steering - previous_steering
    )

    if steering_delta > MAX_STEERING_DELTA:

        excess_delta = (
            steering_delta - MAX_STEERING_DELTA
        )

        damping_factor = max(
            0.4,
            1.0 - excess_delta * STABILIZER_STRENGTH
        )

        final_steering *= damping_factor

        stabilizer_active = True

    previous_steering = final_steering

    # =====================================
    # SEND CONTROLS
    # =====================================

    controller.update_controls(
        steering=final_steering,
        throttle=final_throttle,
        brake=final_brake
    )

    # =====================================
    # UI
    # =====================================

    frame_end = time.perf_counter()
    fps = 1.0 / max(frame_end - prev_frame_time, 1e-6)
    prev_frame_time = frame_end

    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (20, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(processed_frame, f"Infer: {inference_latency_ms:.1f}ms", (160, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    mode_text = (
        "AI ASSIST"
        if assist_mode
        else "MANUAL"
    )

    mode_color = (
        (0, 255, 255)
        if assist_mode
        else (0, 255, 0)
    )

    cv2.putText(
        processed_frame,
        f"Mode: {mode_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        mode_color,
        2
    )

    cv2.putText(
        processed_frame,
        f"Steering: {final_steering:.2f}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Throttle: {final_throttle:.2f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        processed_frame,
        f"Brake: {final_brake:.2f}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 100, 255),
        2
    )

    # =====================================
    # AI PREDICTIONS DISPLAY
    # =====================================

    cv2.putText(
        processed_frame,
        f"AI Steering: {ai_steering:.2f}",
        (850, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        processed_frame,
        f"AI Throttle: {ai_throttle:.2f}",
        (850, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        processed_frame,
        f"AI Brake: {ai_brake:.2f}",
        (850, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # =====================================
    # STABILIZER VISUALIZATION
    # =====================================

    if stabilizer_active:

        cv2.putText(
            processed_frame,
            "STEERING STABILIZER ACTIVE",
            (300, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

    # =====================================
    # DISPLAY WINDOW
    # =====================================

    cv2.imshow(
        "AI Gesture Racing Controller",
        processed_frame
    )

    # =====================================
    # KEYBOARD CONTROLS
    # =====================================

    key = cv2.waitKey(1)

    # Quit
    if key == ord('q'):
        break

    # Toggle AI Assist
    elif key == ord('a'):

        assist_mode = not assist_mode

        print(
            f"\nAI Assist Mode: {assist_mode}"
        )

# =========================================
# CLEANUP
# =========================================

cap.release()
cv2.destroyAllWindows()