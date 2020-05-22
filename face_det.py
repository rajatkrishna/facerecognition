import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import dlib

image_file = "images/20200520_094328.jpg"

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return [x, y, w, h]

def display(img, n, bb_points):
    
    img_rect = img.copy()
    for i in range(n):
        x, y, w, h = bb_points[i]
        r, g, b = np.random.choice(256, 3)
        img_rect = cv2.rectangle(img_rect, (x, y), (x + w, y + h), (r, g, b))
    
    cv2.imshow('Detected faces', img_rect)

img = cv2.imread(img_file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


face_detector = dlib.get_frontal_face_detector()

faces, scores, idx = detector.run(gray, 1, 0.25)

print("{} faces found in image {}".format(len(faces), i + 1))
bb_points = []
for face_rect in faces:
    points = rect_to_bb(face_rect)
    bb_points.append(points)

display(img_copy, len(faces), bb_points)

    

                                       