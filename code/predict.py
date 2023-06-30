from pathlib import Path
import numpy as np
from keras.models import load_model
import cv2
from PIL import Image

import json
face_cascade = cv2.CascadeClassifier(str(Path(__file__).parents[1] / 'opencv/haarcascades/haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(str(Path(__file__).parents[1] / 'opencv/haarcascades/haarcascade_eye.xml'))

def get_cropped_image_if_2_eyes():
    img = cv2.imread("img.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

def find_key_by_value(dictionary, value):
    keys = []
    for key, val in dictionary.items():
        if val == value:
            keys.append(key)
    return keys


def predict():
    # Load the model from the saved file
    model = load_model(str(Path(__file__).parents[1] / 'model/image_classifier.h5'))

    #Image Preprocessing
    cropped_image_no_2_eyes = get_cropped_image_if_2_eyes()
    image_fromarray = Image.fromarray(cropped_image_no_2_eyes, 'RGB')
    resize_image = image_fromarray.resize((30, 30))
    x = np.array(resize_image)
    x = x / 255
    x = x.reshape((1, 30, 30, 3))

    # Prediction
    output = model.predict(x)
    probabilities = [np.round(x * 100,4) for x in output[0]]
    class_names = ["Elon Musk","Kiran Mazumdar Shaw","Jeff Bezos","Mark Zuckerberg","Falguni Nayar"]
    pred = np.argmax(output, axis=1)
    return pred,probabilities,class_names
