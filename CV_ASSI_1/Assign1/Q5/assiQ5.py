import cv2
import numpy as np
from PIL import Image

img1 = cv2.imread("walk_1.jpg")
imgshow = Image.open("walk_1.jpg")
imgshow.show()

img2 = cv2.imread("walk_2.jpg")
imgshow = Image.open("walk_2.jpg")
imgshow.show()

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

gray_sub = np.subtract(gray1,gray2)

cv2.imwrite("Q5_walk1_gray.jpg",gray1)
imgshow = Image.open("Q5_walk1_gray.jpg")
imgshow.show()

cv2.imwrite("Q5_walk2_gray.jpg",gray2)
imgshow = Image.open("Q5_walk2_gray.jpg")
imgshow.show()

cv2.imwrite("Q5_sub_gray.jpg",gray_sub)
imgshow = Image.open("Q5_sub_gray.jpg")
imgshow.show()

