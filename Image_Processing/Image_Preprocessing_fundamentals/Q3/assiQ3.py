import cv2
import numpy as np
from PIL import Image
import random
from scipy.signal import convolve2d




img1 = cv2.imread("Noisyimage1.jpg")
img2 = cv2.imread("Noisyimage2.jpg")


filterd_med_N1 = cv2.medianBlur(img1,5)
cv2.imwrite("Q3_N1_med.jpg",filterd_med_N1)
imgshow = Image.open("Q3_N1_med.jpg")
imgshow.show()



filterd_mean_N1 = cv2.blur(img1,(5,5))
cv2.imwrite("Q3_N1_mean.jpg",filterd_mean_N1)
imgshow = Image.open("Q3_N1_mean.jpg")
imgshow.show()


filterd_med_N2 = cv2.medianBlur(img2,5)
cv2.imwrite("Q3_N2_med.jpg",filterd_med_N2)
imgshow = Image.open("Q3_N2_med.jpg")
imgshow.show()



filterd_mean_N2 = cv2.blur(img2,(5,5))
cv2.imwrite("Q3_N2_mean.jpg",filterd_mean_N2)
imgshow = Image.open("Q3_N2_mean.jpg")
imgshow.show()
