import cv2
import numpy as np
import time
import hand_tracking_module as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
wCam, hCam = 1288, 728
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon=0.75)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
while True:
  success, img = cap.read()
  img = detector.findHands(img, draw=True)
  lmList = detector.findPosition(img, draw=False)
  if len(lmList) != 0: 
    x1, y1 = lmList[4][1], lmList[4][2]
    x2, y2 = lmList[8][1], lmList[8][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (x1, y1), 15, (0, 0, 0), cv2.FILLED)
    cv2.circle(img, (x2, y2), 15, (0, 0, 0), cv2.FILLED)
    cv2.circle(img, (cx, cy), 15, (0, 0, 0), cv2.FILLED)
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 3)
    length = math.hypot(x2 - x1, y2 - y1)
    length = np.clip(length, 50, 300)
    vol = np.log(0.004*length - 0.2)/np.log(1.072)
    vol = np.clip(vol, -65, 0)
    volBar = np.interp(length, [50, 300], [400, 150])
    volPer = np.interp(length, [50, 300], [0, 100])
    volume.SetMasterVolumeLevel(vol, None)
  cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 255), 3)
  cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 255), cv2.FILLED)
  cTime = time.time()
  fps = 1 / (cTime - pTime)
  cv2.putText(img, f'FPS: {str(int(fps))}', (1060, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
  pTime = cTime
  cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
  cv2.imshow("Image", img)
  cv2.waitKey(1)