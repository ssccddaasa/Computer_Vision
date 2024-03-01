import cv2
import numpy as np
from PIL import Image
import random
from scipy.signal import convolve2d


img = cv2.imread("Q1.jpg",0)
#Q1-1 show image.
imgshow = Image.open("Q1.jpg")
imgshow.show()


#Q1-2 image transformation

gamma4 = np.array(255*(img/255)** 0.4, dtype="uint8")

cv2.imwrite("Q1_transformed0.4.jpg",gamma4)

imgshow = Image.open("Q1_transformed0.4.jpg")
imgshow.show()



#Q1-3 image  Gaussian noise
print(img.shape)
gauss = np.zeros((225,225,3), dtype=np.uint8)
cv2.randn(gauss,0,np.sqrt(40))
gauss = (gauss*0.5).astype(np.uint8)



img_gus = img + gauss

cv2.imwrite("Q1_gaussian.jpg",img_gus)

imgshow = Image.open("Q1_gaussian.jpg")
imgshow.show()


#Q1-4 mean filter

m = 225
n = 225

mask = np.ones([5,5],dtype=int)
mask = mask/25

img_mean = np.zeros([m,n])

for i in range(1,m-2):
    for j in range(1,n-2):
        avg1 = img[i-2, j-2]*mask[0, 0]+img[i-2, j-1]*mask[0, 1]+img[i-2, j]*mask[0, 2]+img[i-2, j+1]*mask[0, 3]+ img[i-2, j+2]*mask[0, 4]
        avg2 = img[i-1, j-2]*mask[1, 0]+img[i-1, j-1]*mask[1, 1]+img[i-1, j]*mask[1, 2]+img[i-1, j+1]*mask[1, 3]+ img[i-1, j+2]*mask[1, 4]
        avg3 = img[i, j-2]*mask[2, 0]+img[i, j-1]*mask[2, 1]+img[i, j]*mask[2, 2]+img[i, j+1]*mask[2, 3]+ img[i, j+2]*mask[2, 4]
        avg4 = img[i+1, j-2]*mask[3, 0]+img[i+1, j-1]*mask[3, 1]+img[i+1, j]*mask[3, 2]+img[i+1, j+1]*mask[3, 3]+ img[i+1, j+2]*mask[3, 4]
        avg5 = img[i+2, j-2]*mask[4, 0]+img[i+2, j-1]*mask[4, 1]+img[i+2, j]*mask[4, 2]+img[i+2, j+1]*mask[4, 3]+ img[i+2, j+2]*mask[4, 4]

        avgtot = avg1 + avg2 +avg3 + avg4 + avg5
        img_mean[i,j] = avgtot


img_mean = img_mean.astype(np.uint8)

cv2.imwrite("Q1_mean.jpg",img_mean)

imgshow = Image.open("Q1_mean.jpg")
imgshow.show()





#Q1-5 salt and pepper noice
img2 = cv2.imread("Q1.jpg")
density = (m*n) // 10

for i in range(density//2):

    y = random.randint(0,224)
    x = random.randint(0,224)
    img2[y][x] = 255

for i in range(density//2):

    y = random.randint(0,224)
    x = random.randint(0,224)
    img2[y][x] = 0

cv2.imwrite("Q1_salt.jpg",img2)
imgshow = Image.open("Q1_salt.jpg")
imgshow.show()


img3 = cv2.imread("Q1_salt.jpg")
filterd_med = cv2.medianBlur(img3,7)
cv2.imwrite("Q1_medsalt.jpg",filterd_med)
imgshow = Image.open("Q1_medsalt.jpg")
imgshow.show()

# Q1-6 salt_mean.

filterd_mean = cv2.blur(img3,(7,7))
cv2.imwrite("Q1_meansalt.jpg",filterd_mean)
imgshow = Image.open("Q1_meansalt.jpg")
imgshow.show()


# Q1-7 soble

def soble(img):
    sob_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=int)
    img_sob_x =convolve2d(img,sob_x,mode="same",boundary="symm")

    sob_y = np.array([[-1,2,-1],[0,0,0],[1,2,1]],dtype=int)
    img_sob_y = convolve2d(img,sob_y,mode="same",boundary="symm")

    img_sob = np.sqrt(img_sob_x**2 + img_sob_y**2)

   
    img_sobr = img_sob
    img_sobr = img_sobr.astype(np.uint8)
    cv2.imwrite("Q1_soble.jpg",img_sobr)
    imgshow = Image.open("Q1_soble.jpg")
    imgshow.show()



img = cv2.imread("Q1.jpg",0)
print(img.shape)

soble(img)




