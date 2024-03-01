import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


img = cv2.imread("Q_4.jpg",0)
imgshow = Image.open("Q_4.jpg")
imgshow.show()


gx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
gy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

grad_mag = np.sqrt(gx**2 + gy**2)

grad_mag_str = cv2.normalize(grad_mag,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)

cv2.imwrite("Q4_soble.jpg",grad_mag_str)
imgshow = Image.open("Q4_soble.jpg")
imgshow.show()

mag_his = cv2.calcHist([grad_mag_str.astype(np.float32)],[0],None,[256],[0,256])

plt.plot(mag_his, color='black')
plt.title('Histogram of Gradient Magnitude')
plt.xlabel('Gradient Magnitude Intensity')
plt.ylabel('Frequency')
plt.show()



grad_ori = np.arctan2(gy,gx)


plt.imshow(grad_ori, cmap='hsv')
plt.colorbar()
plt.title('Gradient Orientation')
plt.axis('off')
plt.show()

grad_ori[grad_ori<0] += 2*np.pi

ori_hist, bins = np.histogram(grad_ori, bins=255,range=(0,2*np.pi))


plt.bar(bins[:-1], ori_hist, width=2 * np.pi / 255, align='edge')
plt.title('Histogram of Gradient Orientation')
plt.xlabel('Orientation (radians)')
plt.ylabel('Frequency')
plt.show()
