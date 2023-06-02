# This module take frames with detected books and 
# removes the shadow from it and converts it to Gray and Binary
# and then the Tesseract OCR detects and reads the words
# Tesseract is not ideal for OCR so we have also tested with some deep learning models like Deep Text

from PIL import Image
import pytesseract
import argparse
import cv2
import os
from gtts import gTTS
import numpy as np

# load the example image
image = cv2.imread("C:/WPI/lego/croppedUnblurred.jpg")
# ------------------------------------------------------------------------
# Shadow removal

rgb_planes = cv2.split(image)

result_planes = []
result_norm_planes = []
i = 0

for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))

    bg_img = cv2.medianBlur(dilated_img, 21)

    diff_img = 255 - cv2.absdiff(plane, bg_img)
    
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

image = cv2.merge(result_norm_planes)
cv2.imwrite("C:/WPI/lego/del/image{}.jpg".format(i), image)

# ------------------------------------------------------------------------
# OCR

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
cv2.imwrite("C:/WPI/lego/med.jpg", gray)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

text = pytesseract.image_to_string(Image.open(filename))

tts = gTTS(text, lang='en')
tts.save('C:/WPI/lego/speech.mp3')   

os.remove(filename)
print(text)

# # show the output images
# cv2.imshow("Image", image)
# cv2.imshow("Output", gray)
# cv2.waitKey(0)
