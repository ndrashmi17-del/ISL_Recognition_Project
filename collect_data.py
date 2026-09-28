import cv2
import mediapipe as mp
import os
import csv

# Ask for sign name
sign_name = input("Enter sign name: ")

# Create folder if it doesn't exist
folder_path = os.path.join("dataset", sign_name)
os.makedirs(folder_path, exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

sample_count = 0

print("Press 's' to save a sample")
print("Press 'q' to quit")

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            keypoints = []

            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):

                sample_count += 1

                filename = os.path.join(folder_path, f"{sample_count}.csv")

                with open(filename, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(keypoints)

                print(f"Saved sample {sample_count}")

            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.putText(
        frame,
        f"Samples: {sample_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Dataset", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()