import pickle
import numpy as np
import cv2
import pandas as pd

propability_needed = 0.9

labels = pd.read_csv('german-traffic-signs-1/signnames.csv')
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)
cap.set(3, 1280) # zmiena szerokosci
cap.set(4, 720) # zmiana wysokosci
cap.set(10, 180) # zmiana jasnosci

pickle_in = open("myTraffic_sign_model.p", "rb")
model = pickle.load(pickle_in)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def preprocesing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

print(labels)

