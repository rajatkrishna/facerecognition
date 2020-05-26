import numpy as np 
import cv2
from face_det import face_det

face_detector = face_det()
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    frame = face_detector.get_output(frame)
    cv2.imshow('Frames', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows