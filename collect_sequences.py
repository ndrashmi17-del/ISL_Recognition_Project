import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# ============================================================
# SETTINGS
# ============================================================
SEQUENCES = 10          # how many sequences per sign
FRAMES = 30
DATA_PATH = "dataset"

VALID_SIGNS = [
    "Hello", "Thank You", "Help", "Yes", "No",
    "Please", "Sorry", "Good Morning", "Good Night", "How Are You",
    "Welcome", "Come", "Go", "Stop", "Eat",
    "Drink", "Water", "Home", "Friend", "Love"
]

FACE_INDICES = [1, 10, 33, 61, 152, 199, 263, 291, 454, 467]

# ============================================================
# SELECT SIGN
# ============================================================
print("\n" + "="*50)
print("       SIGNSPEAK DATASET RECORDER")
print("="*50 + "\n")

for i, sign in enumerate(VALID_SIGNS, 1):
    print(f"{i:2d}. {sign}")

print("\n" + "="*50)
ACTION = input("Enter sign name exactly as shown above: ").strip()

if ACTION not in VALID_SIGNS:
    print("\nERROR: Invalid sign name.")
    sys.exit()

ACTION_PATH = os.path.join(DATA_PATH, ACTION)
os.makedirs(ACTION_PATH, exist_ok=True)

# ============================================================
# CHECK EXISTING
# ============================================================
existing = []
for f in os.listdir(ACTION_PATH):
    if f.startswith("sequence_") and f.endswith(".npy"):
        try:
            num = int(f.replace("sequence_", "").replace(".npy", ""))
            existing.append(num)
        except:
            pass

existing = sorted(existing)
print(f"\nSign: {ACTION}")
print(f"Existing sequences: {len(existing)}/{SEQUENCES}")

if len(existing) >= SEQUENCES:
    print("This sign already has enough sequences.")
    sys.exit()

missing = [i for i in range(1, SEQUENCES + 1) if i not in existing]
print("Still need:", missing)

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_landmarks(results):
    # Left hand
    left = np.zeros(63, dtype=np.float32)
    if results.left_hand_landmarks:
        left = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                        dtype=np.float32).flatten()

    # Right hand
    right = np.zeros(63, dtype=np.float32)
    if results.right_hand_landmarks:
        right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                         dtype=np.float32).flatten()

    # Pose
    pose = np.zeros(99, dtype=np.float32)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
                        dtype=np.float32).flatten()

    # Face
    face = np.zeros(30, dtype=np.float32)
    if results.face_landmarks:
        vals = []
        for idx in FACE_INDICES:
            lm = results.face_landmarks.landmark[idx]
            vals.extend([lm.x, lm.y, lm.z])
        face = np.array(vals, dtype=np.float32)

    return np.concatenate([left, right, pose, face]).astype(np.float32)

# ============================================================
# CAMERA + RECORDING
# ============================================================
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    sys.exit()

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    for seq_num in missing:
        print(f"\n→ Recording Sequence {seq_num}/{SEQUENCES} for '{ACTION}'")
        print("Press 'S' to start recording | 'Q' to quit")

        # Wait for S
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, "PRESS S TO START | Q TO QUIT", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Sign: {ACTION} | Seq: {seq_num}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SignSpeak Recorder", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

        # Record 30 frames
        sequence_data = []
        for f in range(FRAMES):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Draw
            if results.pose_landmarks:
                mp_draw.draw_landmarks(display, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(display, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(display, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_landmarks(results)
            if keypoints.shape[0] != 255:
                print("ERROR: Feature size wrong!")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

            sequence_data.append(keypoints)

            cv2.putText(display, f"Frame: {f+1}/{FRAMES}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Sign: {ACTION}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SignSpeak Recorder", display)
            cv2.waitKey(1)

        # Save
        arr = np.array(sequence_data, dtype=np.float32)
        save_path = os.path.join(ACTION_PATH, f"sequence_{seq_num:03d}.npy")
        np.save(save_path, arr)
        print(f"✓ Saved → {save_path} | Shape: {arr.shape}")

cap.release()
cv2.destroyAllWindows()
print("\nData collection finished successfully!")