import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import dlib

image_file = "images/download.jpeg"

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return [x, y, w, h]

def display(img, n, bb_points):
    
    img_rect = np.copy(img)
    for i in range(n):
        x, y, w, h = bb_points[i]
        r, g, b = np.random.choice(256, 3)
        img_rect = cv2.rectangle(img_rect, (x, y), (x + w, y + h), (r.item(), g.item(), b.item()), thickness = 2)
    
    img_rect = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    cv2.imshow("Detected faces", img_rect)
    cv2.waitKey()

img = cv2.imread(image_file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


face_detector = dlib.get_frontal_face_detector()

faces, scores, idx = face_detector.run(img, 1, 0.25)

print("{} faces found in image {}".format(len(faces), image_file))
bb_points = []
for face_rect in faces:
    points = rect_to_bb(face_rect)
    bb_points.append(points)

if len(faces) > 0:
    display(img_copy, len(faces), bb_points)

    

                                       