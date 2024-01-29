import cv2
import numpy as np
import time
import hand_tracking_module as htm
import math
wCam, hCam = 1288, 728
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.75, maxHands=4)
tipIds = [4, 8, 12, 16, 20]
while True:
  success, img = cap.read()
  img = detector.findHands(img, draw=True)
  lmList = detector.findPosition(img, both_hands=True, draw=False)
  if len(lmList) != 0:
    totalFingers = 0
    for handlmList in lmList:
      fingers = 0
      for id in range(1, 5):
        if handlmList[tipIds[id]][2] < handlmList[tipIds[id] - 2][2]:
          fingers += 1
      if handlmList[4][1] < handlmList[20][1]:
        if handlmList[tipIds[0]][1] < handlmList[tipIds[0] - 2][1]:
          fingers += 1
      else:
        if handlmList[tipIds[0]][1] > handlmList[tipIds[0] - 2][1]:
          fingers += 1
      totalFingers += fingers
    if totalFingers >= 10:
      cv2.rectangle(img, (20, 225), (280, 425), (0, 255, 0), cv2.FILLED)
    else:
      cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  pTime = cTime
  cv2.putText(img, f'FPS: {str(int(fps))}', (1060, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
  cv2.imshow("Image", img)
  cv2.waitKey(1)