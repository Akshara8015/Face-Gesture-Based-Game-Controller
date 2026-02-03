# -------------------------------
# FACE POINTS COORDINATES
# -------------------------------
def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])

# -------------------------------
# PREDICTING HEAD POSTURE
# -------------------------------
def classify_head_pose(yaw_ratio, pitch_ratio, roll_angle):
    if roll_angle > 17:
        return "TILT RIGHT"
    elif roll_angle < -22:
        return "TILT LEFT"
    elif pitch_ratio > 1.60:
        return "DOWN"
    elif pitch_ratio < 0.95:
        return "UP"
    elif yaw_ratio > 1.70:
        return "RIGHT"
    elif yaw_ratio < 0.50:
        return "LEFT"
    else:
        return "NEUTRAL"

# -------------------------------
# MARKING POINTS ON FACE
# -------------------------------
def draw_points_on_face(latest_result, width, height):
    face_landmarks = latest_result.face_landmarks[0]

    nose = get_point(face_landmarks, 1, width, height)
    chin = get_point(face_landmarks, 152, width, height)
    left_eye = get_point(face_landmarks, 33, width, height)
    right_eye = get_point(face_landmarks, 263, width, height)
    forehead = get_point(face_landmarks, 10, width, height)

    # for p in [nose, forehead]:
    for p in [nose, chin, left_eye, right_eye, forehead]:
        cv2.circle(frame, tuple(p.astype(int)), 4, (0,255,0), -1)

    # cv2.line(frame, (frame_width // 2, 0), (frame_width // 2, frame_height), (0, 255, 0), 2)  # Vertical line
    # cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2), (0, 255, 0), 2)  # Horizontal line
    # ---- YAW (LEFT / RIGHT) ----
    yaw_ratio = (nose[0] - left_eye[0]) / (right_eye[0] - nose[0] + 1e-6)

    # ---- PITCH (UP / DOWN) ----
    pitch_ratio = (nose[1] - forehead[1]) / (chin[1] - nose[1] + 1e-6)

    # ---- ROLL (TILT) ----
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll_angle = np.degrees(np.arctan2(dy, dx))

    posture = classify_head_pose(yaw_ratio, pitch_ratio, roll_angle)

    # ---- DISPLAY ----
    cv2.putText(frame, f"Yaw: {yaw_ratio:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Pitch: {pitch_ratio:.2f}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"Roll: {roll_angle:.1f}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    cv2.putText(frame, f"POSTURE: {posture}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    return posture

# -------------------------------
# GAME ACTION MAPPING
# -------------------------------
def perform_action(label):
    global last_action_time
    now = time.time()

    if now - last_action_time < COOLDOWN:
        return
    # print("label",label)
    if label == "LEFT":
        pyautogui.keyDown("left"); pyautogui.keyUp("left")

    elif label == "RIGHT":
        pyautogui.keyDown("right"); pyautogui.keyUp("right")

    elif label == "UP":
        pyautogui.keyDown("up"); pyautogui.keyUp("up")

    elif label == "DOWN":
        pyautogui.keyDown("down"); pyautogui.keyUp("down")

    elif label == "TILT LEFT":
        pyautogui.keyDown("z"); pyautogui.keyUp("z")

    elif label == "TILT RIGHT":
        pyautogui.keyDown("x"); pyautogui.keyUp("x")

    last_action_time = now
    # print(f"[ACTION] {label.upper()}")

# -------------------------------
# STABILIZATION LOGIC
# -------------------------------

def smooth_prediction(pred):
    pred_buffer.append(pred)

    if len(pred_buffer) < BUFFER_SIZE:
        return NEUTRAL

    return Counter(pred_buffer).most_common(1)[0][0]

