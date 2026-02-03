# Face-Gesture-Based-Game-Controller

## Head Pose Controlled Game Using Computer Vision
Control games hands-free using just your head movements!
This project uses real-time facial landmark detection and geometric head pose estimation to map head movements to keyboard actions — no deep learning, no training required.

## Features
- Real-time webcam-based face tracking
- Geometry-based head pose estimation (Yaw, Pitch, Roll)
- Detects 7 head postures: Left, Right, Up, Down, Tilt Left, Tilt Right, Neutral

## Prediction smoothing & cooldown logic for stability
- Hands-free game control using keyboard automation
- Low-latency, model-free solution

## How It Works

### 1. Face Landmark Detection
- Uses MediaPipe Face Landmarker (468 facial landmarks)
- Extracts key points: Nose, Chin, Forehead, Left Eye, Right Eye

### 2. Head Pose Estimation (Geometry-Based)
- Yaw → Left / Right movement
- Pitch → Up / Down movement
- Roll → Tilt Left / Tilt Right

### 3. Stabilization & Safety Logic
- Temporal smoothing using prediction buffers
- Cooldown mechanism to prevent accidental repeated actions

### 4. Game Control
- Mapped head postures to keyboard inputs using PyAutoGUI
- Works with keyboard-controlled games (e.g., Temple Run, Subway Surfers on emulator)

## Tech Stack
- Language: Python
- Computer Vision: OpenCV, MediaPipe
- Math & Geometry: NumPy, Trigonometry
- Automation: PyAutoGUI
- System: Webcam, Real-time inference



