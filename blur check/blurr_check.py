import cv2
import numpy as np

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
video = cv2.VideoCapture('C:/WPI/lego/Demo//result.mp4')
temp_var = 0
i = 1
# Read the image
ret, img = video.read()
while(ret):
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # Find the laplacian of this image and
    # # calculate the variance
    var = cv2.Laplacian(grey, cv2.CV_64F).var()
    
    # # if variance is less than the set threshold
    # # image is blurred otherwise not
    # print(var)
    if(var>temp_var):
        temp_var = var
        temp_img = img
    img = cv2.filter2D(img, -1, kernel)
    cv2.imwrite("C:/WPI/lego/del/unblurred{}.jpg".format(var), img)
    i+=1
    ret, img = video.read()

temp_img = cv2.filter2D(temp_img, -1, kernel)
cv2.imwrite("C:/WPI/lego/unblurred.jpg", temp_img)


