
import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic()

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

os.makedirs("dataset/landmarks", exist_ok=True)

frame_count = 0

while camera.isOpened():

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = holistic.process(rgb)

    landmarks=[]

    if results.left_hand_landmarks:

        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x,lm.y,lm.z])

    else:
        landmarks.extend([0]*63)

    if results.right_hand_landmarks:

        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x,lm.y,lm.z])

    else:
        landmarks.extend([0]*63)

    np.save(f"dataset/landmarks/frame_{frame_count}.npy", np.array(landmarks))

    cv2.imshow("Landmark Recorder",frame)

    frame_count+=1

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

camera.release()
cv2.destroyAllWindows()