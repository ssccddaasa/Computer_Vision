import cv2
import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def myImgfil(img,kernal):
    km,kn = kernal.shape

    pad_t = km//2
    pad_b = km - pad_t -1
    pad_l = kn // 2
    pad_r = kn - pad_l -1
    

    pad_img = np.pad(img,pad_width=((pad_t,pad_b),(pad_l,pad_r)),mode="constant")

    out = convolve2d(pad_img,kernal,mode="valid")

    return out

def gauss(sig):
    size = 2*sig + 1
    kernal = np.zeros((size,size))
    cen = size // 2


    for i in range(size):
        for j in range(size):
            x = i - cen
            y = j - cen

            kernal[i,j] = np.exp(-(x**2 + y**2) / (2* sig**2))
    
    kernal = kernal / (2*np.pi * sig**2)
    kernal = kernal / kernal.sum()

    return kernal



# Q2_mean
img1 = cv2.imread("House1.jpg",0)
img2 = cv2.imread("House2.jpg",0)

avg_kernal_3 = np.ones([3,3],dtype=int)
avg_kernal_3 = avg_kernal_3 /9

avg_kernal_5 = np.ones([5,5],dtype=int)
avg_kernal_5 = avg_kernal_5 /25

H1_avg_3 = myImgfil(img1,avg_kernal_3)
cv2.imwrite("Q2_H1_avg_3.jpg",H1_avg_3)
imgshow = Image.open("Q2_H1_avg_3.jpg")
imgshow.show()


H1_avg_5 = myImgfil(img1,avg_kernal_5)
cv2.imwrite("Q2_H1_avg_5.jpg",H1_avg_5)
imgshow = Image.open("Q2_H1_avg_5.jpg")
imgshow.show()


H2_avg_3 = myImgfil(img2,avg_kernal_3)
cv2.imwrite("Q2_H2_avg_3.jpg",H2_avg_3)
imgshow = Image.open("Q2_H2_avg_3.jpg")
imgshow.show()


H2_avg_5 = myImgfil(img2,avg_kernal_5)
cv2.imwrite("Q2_H2_avg_5.jpg",H2_avg_5)
imgshow = Image.open("Q2_H2_avg_5.jpg")
imgshow.show()

# Q_2 gaussian

for sigma in range (1,4):
    gauss_s1 = gauss(sigma)

    H1_gauss_s1 = myImgfil(img1,gauss_s1)
    cv2.imwrite("Q2_H1_gauss_s"+str(sigma)+".jpg",H1_gauss_s1)
    imgshow = Image.open("Q2_H1_gauss_s"+str(sigma)+".jpg")
    imgshow.show()

    H2_gauss_s1 = myImgfil(img2,gauss_s1)
    cv2.imwrite("Q2_H2_gauss_s"+str(sigma)+".jpg",H2_gauss_s1)
    imgshow = Image.open("Q2_H2_gauss_s"+str(sigma)+".jpg")
    imgshow.show()


# Q2_ soble

def soble(img,h):

    pad_img = np.pad(img,pad_width=((1,1),(1,1)),mode="constant")
    sob_x = np.array([[-1,2,-1],[0,0,0],[1,2,1]],dtype=int)
    img_sob_x =convolve2d(pad_img,sob_x,mode="same",boundary="symm")

    sob_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=int)
    img_sob_y = convolve2d(pad_img,sob_y,mode="same",boundary="symm")

    img_sob = np.sqrt(img_sob_x**2 + img_sob_y**2)

   
    img_sobr = img_sob
    img_sobr = img_sobr.astype(np.uint8)
    cv2.imwrite("Q2_"+h+"_soble.jpg",img_sobr)
    imgshow = Image.open("Q2_"+h+"_soble.jpg")
    imgshow.show()




soble(img1,"H1")
soble(img2,"H2")

# Q2_prewitt

def prewitt(img,h):

    pad_img = np.pad(img,pad_width=((1,1),(1,1)),mode="constant")
    pre_x = np.array([[1,1,1],[0,0,0],[-1,-1,1]],dtype=int)
    img_pre_x =convolve2d(pad_img,pre_x,mode="same",boundary="symm")

    pre_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
    img_pre_y = convolve2d(pad_img,pre_y,mode="same",boundary="symm")

    img_pre = np.sqrt(img_pre_x**2 + img_pre_y**2)

   
    img_prew = img_pre
    img_prew = img_prew.astype(np.uint8)
    cv2.imwrite("Q2_"+h+"_prew.jpg",img_prew)
    imgshow = Image.open("Q2_"+h+"_prew.jpg")
    imgshow.show()





prewitt(img1,"H1")
prewitt(img2,"H2")


