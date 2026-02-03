import numpy as np
import mediapipe as mp
import pyautogui
from collections import deque, Counter
import cv2
import time
import helper_file

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_faces=1)

detector = FaceLandmarker.create_from_options(options)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

BUFFER_SIZE = 4
COOLDOWN = 0.08   # seconds
NEUTRAL = "NEUTRAL"

# -------------------------------
# STABILIZATION BUFFERS
# -------------------------------
pred_buffer = deque(maxlen=BUFFER_SIZE)
last_action_time = 0
prev_label = NEUTRAL

cap = cv2.VideoCapture(0)
start_time = time.time()

with (FaceLandmarker.create_from_options(options) as detector):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw quadrant lines
        frame_height, frame_width, _ = frame.shape

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int((time.time() - start_time) * 1000)
        detector.detect_async(mp_image, timestamp_ms)

        if latest_result and latest_result.face_landmarks:
            raw_label = helper_file.draw_points_on_face(latest_result, frame_width, frame_height)
            stable_label = helper_file.smooth_prediction(raw_label)
            if stable_label != prev_label and stable_label != NEUTRAL:
                helper_file.perform_action(stable_label)

            prev_label = stable_label
            if not latest_result or not latest_result.face_landmarks:
                prev_label = "neutral"
                continue


        cv2.imshow("AKSHARA JAIN", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
