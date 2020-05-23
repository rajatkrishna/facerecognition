import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from face_aligner import face_aligner

class face_det():

    def __init__(self, image_file = "images/download.jpeg", 
             keypoints_predictor = "keypoints_model/shape_predictor_68_face_landmarks.dat",
             threshold = 0.25,
             display = False,
             display_aligned = False):

        self.image_file = image_file
        self.keypoints_predictor = keypoints_predictor
        self.threshold = threshold

        face_detector = dlib.get_frontal_face_detector()

        self.load_img()
        print("Image loaded successfully...")
        self.find_faces(face_detector = face_detector,  
                        display = display, 
                        display_aligned = display_aligned)

    def rect_to_bb(self, rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        return [x, y, w, h]

    def display(self, img, n, bb_points, keypoints):
        
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

    def keypoints(self, img, rects, n):

        predictor = dlib.shape_predictor(self.keypoints_predictor)

        keypoints = np.zeros((n, 68, 2), dtype = int)
        for j, rect in enumerate(rects):
            shape = predictor(img, rect)

            for i in range(68):
                keypoints[j, i] = (shape.part(i).x, shape.part(i).y)

        return keypoints

    def load_img(self):
        self.img = cv2.imread(self.image_file)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img_copy = self.img.copy()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

    def find_faces(self, face_detector, display_aligned, display = False):
    
        faces, _, _ = face_detector.run(self.img, 1, self.threshold)
        self.no_faces = len(faces)
        self.faces = faces

        print("{} face(s) found in image {}".format(self.no_faces, self.image_file))

        if self.no_faces == 0:
            return

        bb_points = []

        for face_rect in self.faces:
            points = self.rect_to_bb(face_rect)
            bb_points.append(points)

        self.bb_points = bb_points
   
        self.face_keypoints = self.keypoints(self.img, faces, self.no_faces)
       
        face_align = face_aligner(img = self.img, 
                                        faces = self.faces, 
                                        keypoints = self.face_keypoints, 
                                        no_faces = self.no_faces, 
                                        face_width = 224)
        
        
        aligned_faces = face_align.get_aligned_faces(display = display_aligned)
 
        if display:

            self.display(img = self.img_copy,
                         n = self.no_faces,
                         bb_points = self.bb_points,
                         keypoints = self.face_keypoints)

    

    


                                       