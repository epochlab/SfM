#!/usr/bin/env python3

from display import Display
from extractor import FeatureExtraction

import cv2
import numpy as np

DATA = '/Users/James/Downloads/Lighthouse.mp4'

W = 4096//2
H = 2160//2
disp = Display(W, H)

F = 1
K = np.array([[F,0,W//2], [0,F,H//2], [0,0,1]])

fe = FeatureExtraction(K)

def process_frame(x):
    img = cv2.resize(x, (W, H))
    features = fe.extract(img)
    for pt1, pt2 in features:
        u1,v1 = fe.denormalise(pt1)
        u2,v2 = fe.denormalise(pt2)
        cv2.circle(img, (u1,v1), color=(0,255,0), radius=1)
        cv2.line(img, (u1,v1), (u2,v2), color=(0,0,255))

    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture(DATA)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break