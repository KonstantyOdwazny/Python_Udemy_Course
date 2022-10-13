import cv2
import numpy as np
from matplotlib import pyplot as plt


    
img = cv2.imread('images/road1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # zeby uzyc matplotlib
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#1.  Define ROI - region zainteresowania 
print(img.shape)
height = img.shape[0]
width = img.shape[1]
roi_verticies = [
    (0,height),
    (width/2, 0),
    (width, height)
]
def region_of_intrest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,)* channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cropp_image = region_of_intrest(img, np.array([roi_verticies], np.int32))


plt.imshow(cropp_image)
plt.show()