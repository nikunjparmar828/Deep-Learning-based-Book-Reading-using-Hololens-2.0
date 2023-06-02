# Only for the testing perpose 
# live stream book detection testing for HoloLens 2.0

import cv2
import time
import numpy as np

import detector
from detector.detector import Detector
from detector.core.utils import draw_bbox

stream = cv2.VideoCapture("https://tkpatel1:tkpatel1234@192.168.0.103/api/holographic/stream/live.mp4?olo=true&pv=true&mic=false&loopback=true")
cv2.namedWindow('live cam', cv2.WINDOW_NORMAL)

# # get frame information
cameraFPS = int(stream.get(cv2.CAP_PROP_FPS))
print("-----------------------------------------------------")
print(cameraFPS)
print("-----------------------------------------------------")
frame_width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0

codec = cv2.VideoWriter_fourcc(*"XVID")

try:
    out = cv2.VideoWriter("C:\WPI\lego\\result.mp4", codec, cameraFPS, (frame_width, frame_height))
except Exception as e:
    print("ERROR : Exception during Camera FPS")
    raise(e)

yoloInference = Detector(tiny=False)

while(i<1200):
    (grabbed, frame) = stream.read()
    detections, image, crop_coords = yoloInference.detect(frame)
    result = np.asarray(image)

    out.write(result)
    for detection in detections:
        classLabel, confidence = detection[0], detection[1]
        if(classLabel=="book" and confidence > 0.8):
            result = np.asarray(image)
            # -------------------------------------------------------------------------------------------
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(result)
            # cv2.imwrite("C:/WPI/lego/img.jpg", image)
    
    # cv2.imwrite("C:/WPI/lego/res/img{}.jpg".format(i), frame)
    
    i+=1

stream.release()
# cv2.destroyAllWindows()