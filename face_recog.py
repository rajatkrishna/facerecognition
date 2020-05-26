from facenet_pytorch import InceptionResnetV1
import torch
import pandas as pd
import pickle

class face_recog():
    
    def __init__(self):

        self.resnet = InceptionResnetV1(pretrained = 'vggface2').eval()

        with open('classifier/SVM.pkl', 'rb') as modfile:
            self.idx2class, self.model = pickle.load(modfile)

                

    def get_name(self, detected_face):
    
        face = torch.Tensor(detected_face)
        face.requires_grad = False
        face = face.transpose(2, 0).unsqueeze(0)

        face_embedding = self.resnet(face)
        
        face_embedding = pd.DataFrame(face_embedding)
        pred = self.model.predict(face_embedding)

        return self.idx2class[pred.item()]


