# Image Processing Fundamentals Assignment

This repository contains solutions to the Image Processing Fundamentals assignment. Below you will find instructions on how to run the code, along with explanations for each question.

## Instructions

1. **Download:** Clone or download the repository to your local machine.
2. **Dependencies:** Ensure you have Python and OpenCV installed on your system.
3. **Run:** Execute the scripts provided for each question.
4. **Review:** Examine the output images and reports generated.

## Solution Overview

### Question 1

- Obtain a grayscale image from the internet meeting specified criteria.
- Apply power law transformation with gamma=0.4.
- Add zero-mean Gaussian noise and discuss the results.
- Apply a 5x5 mean filter to the noisy image.
- Add salt and pepper noise, apply 7x7 median filter, and discuss.
- Apply a 7x7 mean filter.
- Implement Sobel filter manually.

### Question 2

- Implement a function for convolution with padding.
- Test with different kernel sizes and types.
  1. Averaging Kernel (3x3 and 5x5)
  2. Gaussian Kernel (σ = 1, 2, 3)
  3. Sobel Edge Operators
  4. Prewitt Edge Operators
- Discuss the results in the report.

### Question 3

- Apply 5x5 Averaging and Median filters to noisy images.
- Compare and discuss why Median filter works better than averaging.

### Question 4

- Compute gradient magnitude for the provided image using built-in Sobel gradients function.
  1. Stretch the resulting magnitude for better visualization.
  2. Compute the histogram of gradient magnitude.
  3. Compute gradient orientation.
  4. Compute histogram of gradient orientation.

### Question 5

- Load two grayscale images and subtract one from the other.
- Discuss the result obtained.

### Question 6

- Apply Canny edge detector on a given image using OpenCV function "Canny".
- Test different values of 'Threshold'.

## Folder Structure


- **Image_Preprocessing_fundamentals:** Main folder.
  - **Q1:** Solutions for Question 1.
  - **Q2:** Solutions for Question 2.
  - **Q3:** Solutions for Question 3.
  - **Q4:** Solutions for Question 4.
  - **Q5:** Solutions for Question 5.
  - **Q6:** Solutions for Question 6.

## How to Run

- Navigate to each question folder.
- Execute the provided scripts or run the code directly.
- Review the output images and reports generated.

Feel free to reach out if you have any questions or concerns. Good luck!
