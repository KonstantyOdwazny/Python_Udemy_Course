import numpy as np
import cv2

def canny_detect(img):
    blur = cv2.GaussianBlur(img, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny
def roi(img):
    width = img.shape[1]
    height = img.shape[0]
    poligons = np.array([[(0, height), (width/2,height/2),(width,height)]], np.int32)
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,poligons, 255)
    region = cv2.bitwise_and(img, mask)
    return region
image = cv2.imread('images/road_image.jpg') # Zwykly obraz BGR
lines_image = np.copy(image)
gray = cv2.cvtColor(lines_image, cv2.COLOR_RGB2GRAY) # Gray scale image
canny = canny_detect(gray)
region_of_intrest = roi(canny)
cv2.imshow('Image', region_of_intrest)
cv2.waitKey(0)
cv2.destroyAllWindows()