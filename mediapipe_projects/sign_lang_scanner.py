import mediapipe as mp
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('C:/Users/Shrey/Desktop/Computer Vision using MediaPipe/hand_tracking/real-sign-model.h5')
class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self, img, handNo=0, both_hands=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            if both_hands:
                for handLms in self.results.multi_hand_landmarks:
                    handList = []
                    for id, lm in enumerate(handLms.landmark):
                        handList.append([lm.x, lm.y])
                    lmList.append(handList)
            else:
                detectHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(detectHand.landmark):
                    lmList.append([lm.x, lm.y])
        return lmList
detector = handDetector(mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5)
import cv2
import time
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
while True:
  success, img = cap.read()
  img = detector.findHands(img, draw=True)
  lmList = np.array(detector.findPosition(img))
  if np.array(lmList).shape == (21, 2):
    lmList[:, 0] -= lmList[0, 0]
    lmList[:, 1] -= lmList[0, 1]
    predictions = model.predict(np.array([lmList]))
    print(np.argmax(predictions))
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  cv2.putText(img, f'FPS: {str(int(fps))}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
  cv2.imshow("Image", img)
  cv2.waitKey(1)