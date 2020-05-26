from facenet_pytorch import InceptionResnetV1
import sklearn.svm as sksvm
from face_det import face_det
import glob
import torch
import pickle
import re
import pandas as pd
import cv2

train_image_classes = glob.glob("train_images/*")
train_image_classes = [re.findall("(?<=train_images/).*", c)[0] for c in train_image_classes]

idx2class = {}
idx = 0

face_data = pd.DataFrame()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

for image_class in train_image_classes:
    idx2class[idx] = image_class
    train_image_files = glob.glob("train_images/" + image_class + "/*")

    for train_image_file in train_image_files:
        print(train_image_file)
        face_detector = face_det(keypoints_predictor="keypoints_model/shape_predictor_68_face_landmarks.dat",
                                 threshold=0.25)
        img = cv2.imread(train_image_file)
        face_detector.load_img(img)
        _, faces = face_detector.get_aligned_faces()
        for face in faces:
            face = torch.Tensor(face)
            face.requires_grad = False
            face = face.transpose(2, 0).unsqueeze(0)

            face_embedding = resnet(face)

            face_embedding = face_embedding.tolist()

            face_embedding = face_embedding[0] + [idx]
            face_embedding = pd.DataFrame(face_embedding)
            face_data = face_data.append(face_embedding.T)

    idx += 1

if face_data.empty:
    print("Missing images. Please check ./train_images/")
    exit(1)

model = sksvm.SVC()
feat = face_data.drop(face_data.columns[[-1]], axis=1, inplace=False).values
tar = face_data[face_data.columns[[-1]]].values.ravel()

model.fit(feat, tar)

with open('classifier/SVM.pkl', 'wb') as modfile:
    pickle.dump((idx2class, model), modfile)
