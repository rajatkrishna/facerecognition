import cv2
import numpy as np
import dlib
from face_aligner import face_aligner
from face_recog import face_recog


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return [x, y, w, h]


class face_det:

    def __init__(self, keypoints_predictor, threshold=0.25):
        self.keypoints_predictor = keypoints_predictor
        self.threshold = threshold
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_align = face_aligner(face_width=160)
        self.face_rec = face_recog()

    def draw_rect(self, face_points, name):
        x, y, w, h = face_points
        color = (0, 255, 0)
        img_rect = cv2.rectangle(self.img_copy, (x, y), (x + w, y + h), color, thickness=2)
        cv2.putText(img_rect, name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, color)
        img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
        return img_rect

    def keypoints(self, rect):
        predictor = dlib.shape_predictor(self.keypoints_predictor)
        keypoints = np.zeros((68, 2), dtype=int)
        shape = predictor(self.img, rect)

        for i in range(68):
            keypoints[i] = (shape.part(i).x, shape.part(i).y)
        return keypoints

    def load_img(self, img):
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_copy = self.img.copy()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    def get_output(self, img):
        self.load_img(img)
        self.find_faces()
        return self.img_copy

    def get_aligned_faces(self):
        faces, _, _ = self.face_detector.run(self.img, 1, self.threshold)
        aligned_faces = []
        points_list = []
        for face_rect in faces:
            points = rect_to_bb(face_rect)
            face_keypoints = self.keypoints(face_rect)
            aligned_face = self.face_align.align(self.img, face_keypoints)
            aligned_faces.append(aligned_face)
            points_list.append(points)
        return points_list, aligned_faces

    def find_faces(self):
        points_list, aligned_faces = self.get_aligned_faces()
        for (points, aligned_face) in zip(points_list, aligned_faces):
            name = self.face_rec.get_name(aligned_face)
            self.img_copy = self.draw_rect(points, name)
