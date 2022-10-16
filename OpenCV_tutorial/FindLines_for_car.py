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
def draw_lines(img, lines):
    lane_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(lane_img, (x1,y1), (x2,y2), (255,0,0), 8)
    return lane_img
image = cv2.imread('images/road_image.jpg') # Zwykly obraz BGR
copy_img = np.copy(image)
gray = cv2.cvtColor(copy_img, cv2.COLOR_RGB2GRAY) # Gray scale image
canny = canny_detect(gray)
region_of_intrest = roi(canny)
lines = cv2.HoughLinesP(region_of_intrest, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
lines_img = draw_lines(copy_img, lines)
combo_img = cv2.addWeighted(copy_img, 0.8, lines_img, 1, 1)
cv2.imshow('Image', combo_img)
cv2.waitKey(0)
cv2.destroyAllWindows()