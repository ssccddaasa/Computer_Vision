# Content-Based Image Retrieval (CBIR) System

## Background

An image retrieval system is a technology that retrieves relevant images from a database based on user queries. There are two main types of image retrieval systems: text-based and content-based. Text-based systems use keywords or tags to index and search images. Content-based systems analyze the actual content of the image to find similar images. This assignment focuses on developing a Content-Based Image Retrieval (CBIR) system incorporating different color features.

## Objectives

### Overall Objective

Develop a functional CBIR system using different color features and evaluate its performance.

### Specific Objectives

1. Implementation of CBIR System: Develop a Content-Based Image Retrieval (CBIR) system incorporating color histogram and color moment features.
2. Feature Extraction Experimentation: Experiment with the effectiveness of color histogram and color moment in representing image content for retrieval.
3. Evaluate the CBIR system: Define evaluation metrics for assessing the performance of the system (e.g., precision, recall, F1 score), conduct experiments on a benchmark dataset of images, and analyze the effectiveness of different color features and compare their performance.
## Tasks

The assignment contains four main tasks as follows:

1. **Build the CBIR system**: Design and implement a system architecture for image retrieval using color features. Develop functionalities for loading images, extracting features, computing distances, and ranking results.
2. **Implement the CBIR system using Color Histogram**: Experiment with different bin sizes (120, 180, 360) and evaluate using precision, recall, F1 score, and time metrics. Construct a Receiver Operating Characteristic (ROC) curve to measure overall performance.
3. **Experiment with Color Moments**: Implement the CBIR system using Color Moments as an image representation. Experiment with different weighting schemes and additional moments (Median, Mode, Kurtosis).
4. **Improve CBIR system using other image representation techniques**: Experiment with other image representation techniques like Histogram of Oriented Gradients (HOG) and evaluate their performance.

## Dataset

You can experiment with one of the following datasets:
- Wang database: A classic dataset containing 1,000 images from 10 different categories.
- [CBIR Dataset on Kaggle](https://www.kaggle.com/datasets/theaayushbajaj/cbir-dataset)
- Others

## Folder Structure

- `README.md`: This file providing an overview of the project.
- `sol.py`: The Python script containing the implementation of the CBIR system.
- `Images/`: Folder containing the dataset of images (you can use the Wang dataset).
- `databaseCH120.txt`, `databaseCH180.txt`, `databaseCH360.txt`, `databaseCM1.txt`, `databaseCM2.txt`, `databaseCHHOG.txt`: Text files containing feature vectors for the images based on different color features.

## Usage

- Run `sol.py` to train the CBIR system or perform image retrieval tasks.
- Follow the prompts to choose the desired operation (train, CBIR, test).
- Experiment with different algorithms and parameters as specified in the assignment.
