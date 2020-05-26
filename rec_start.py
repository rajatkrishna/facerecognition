import numpy as np 
import cv2
from face_det import face_det
from pathlib import Path

KEYPOINTS_PREDICTOR = Path("./keypoints_model/shape_predictor_68_face_landmarks.dat")
THRESHOLD = 0.25

face_detector = face_det(keypoints_predictor=str(KEYPOINTS_PREDICTOR), threshold=THRESHOLD)
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()

    frame = face_detector.get_output(frame)
    cv2.imshow('Frames', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows