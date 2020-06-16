#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
import sys
from numpy import asarray
from PIL import Image


detecteur = MTCNN ()

def detourage_faces (chemin_image, faces):
    image = plt.imread(chemin_image)
    plt.imshow (image)
    ax = plt.gca()
    for face in faces:
        x1,y1,height,width = face['box']
    
    face_border = Rectangle((x1,y1),height,width, fill = False, color = 'red')
    ax.add_patch(face_border)
    plt.show()


img_envoye=#a compléter avec l'image prise par le telephone
img_base=#a compléter avec l'image de la base de donnée 

image = plt.imread(img_envoye)
faces = detecteur.detect_faces(image)
detourage_faces(img_envoye,faces)


image = plt.imread(img_base)
faces = detecteur.detect_faces(image)
detourage_faces(img_base,faces)


def coupage_faces (image1, image2):
    image = plt.imread(image1)
    detector = MTCNN()
    faces = detector.detect_faces(image)

    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        plt.imshow(face_boundary)
        plt.savefig('recadrage1.jpg')
        
    image = plt.imread(image2)
    faces = detector.detect_faces(image)
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        plt.imshow(face_boundary)
        plt.savefig('recadrage2.jpg')


coupage_faces(img_envoye,img_base)

def get_model_scores(faces):
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50',
      include_top=False,
      pooling='avg')
    
    return model.predict(samples)


image1 = plt.imread('recadrage1.jpg')
image2 = plt.imread('recadrage2.jpg')
faces = [image1, image2]
model_scores = get_model_scores(faces)

if cosine(model_scores[0], model_scores[1]) <= 0.4:
    print(True)
else:
    print(False)

