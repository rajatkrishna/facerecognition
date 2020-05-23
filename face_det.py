import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import dlib
import face_aligner

image_file = "images/download.jpeg"
keypoints_predictor = "keypoints_model/shape_predictor_68_face_landmarks.dat"

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return [x, y, w, h]

def display(img, n, bb_points, keypoints):
    
    img_rect = np.copy(img)
    for i in range(n):
        x, y, w, h = bb_points[i]
        r, g, b = np.random.choice(256, 3, replace = False)
        color = (r.item(), g.item(), b.item())
        img_rect = cv2.rectangle(img_rect, (x, y), (x + w, y + h), color, thickness = 2)
        for (x, y) in keypoints[i]:
            img_rect = cv2.circle(img_rect, (x, y), radius = 1, color = color)
    
    img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    cv2.imshow("Detected faces", img_rect)
    cv2.waitKey()

def keypoints(img, rects, n):
    predictor = dlib.shape_predictor(keypoints_predictor)

    keypoints = np.zeros((n, 68, 2), dtype = int)
    for j, rect in enumerate(rects):
        shape = predictor(img, rect)

        for i in range(68):
            keypoints[j, i] = (shape.part(i).x, shape.part(i).y)

    return keypoints

img = cv2.imread(image_file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


face_detector = dlib.get_frontal_face_detector()

faces, scores, idx = face_detector.run(img, 1, 0.25)
no_faces = len(faces)


print("{} face(s) found in image {}".format(no_faces, image_file))

bb_points = []

for face_rect in faces:
    points = rect_to_bb(face_rect)
    bb_points.append(points)

face_keypoints = keypoints(img, faces, no_faces)

al = face_aligner.face_aligner(img, faces, face_keypoints)

al.display()

if no_faces > 0:
    display(img_copy, no_faces, bb_points, face_keypoints)

    


                                       