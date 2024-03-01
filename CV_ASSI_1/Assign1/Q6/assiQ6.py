import cv2
import numpy as np
from PIL import Image

img = cv2.imread("Q_4.jpg",0)
imgshow = Image.open("Q_4.jpg")
imgshow.show()

img = img.astype(np.uint8)


can_img1 = cv2.Canny(img, 25, 50,apertureSize=3,L2gradient=True)
can_img2 = cv2.Canny(img, 50, 100,apertureSize=3,L2gradient=True)
can_img3 = cv2.Canny(img, 100, 200,apertureSize=3,L2gradient=True)
can_img4 = cv2.Canny(img, 1, 25,apertureSize=3,L2gradient=True)
can_img5 = cv2.Canny(img, 100, 200,apertureSize=5,L2gradient=True)
can_img6 = cv2.Canny(img, 100, 200,apertureSize=3)


cv2.imwrite("Q6_canny_1.jpg",can_img1)
imgshow = Image.open("Q6_canny_1.jpg")
imgshow.show()

cv2.imwrite("Q6_canny_2.jpg",can_img2)
imgshow = Image.open("Q6_canny_2.jpg")
imgshow.show()

cv2.imwrite("Q6_canny_3.jpg",can_img3)
imgshow = Image.open("Q6_canny_3.jpg")
imgshow.show()

cv2.imwrite("Q6_canny_4.jpg",can_img4)
imgshow = Image.open("Q6_canny_4.jpg")
imgshow.show()

cv2.imwrite("Q6_canny_5.jpg",can_img5)
imgshow = Image.open("Q6_canny_5.jpg")
imgshow.show()

cv2.imwrite("Q6_canny_6.jpg",can_img6)
imgshow = Image.open("Q6_canny_5.jpg")
imgshow.show()