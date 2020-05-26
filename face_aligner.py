import cv2
import numpy as np


class face_aligner():

    def __init__(self, face_width=160):
        self.face_width = face_width
        self.face_height = face_width
        self.desired_left_eye = (0.35, 0.35)

    def align(self, img, keypoints):
        left_eye_start = 42
        left_eye_end = 48
        right_eye_start = 36
        right_eye_end = 42
        left_eye_points = keypoints[left_eye_start: left_eye_end, :]
        right_eye_points = keypoints[right_eye_start: right_eye_end, :]
        left_eye_center = left_eye_points.mean(axis=0).astype(int)
        right_eye_center = right_eye_points.mean(axis=0).astype(int)

        (dX, dY) = right_eye_center - left_eye_center
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        desired_right_eye_x = 1 - self.desired_left_eye[0]
        img_dist = np.sqrt(((dX ** 2) + (dY ** 2)))
        des_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.face_width
        scale = des_dist / img_dist
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        tX = self.face_width * 0.5
        tY = self.face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        output = cv2.warpAffine(img, M, (self.face_width, self.face_height))
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output
