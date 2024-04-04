# Vegetable Recognition CNN Model

Dataset: [here](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
Deployment: [here](https://huggingface.co/spaces/irvandhistanly/gc7)

This repository contains code and documentation for a convolutional neural network (CNN) model developed to accurately predict various types of vegetables based on input images. The project involves exploratory data analysis (EDA), model development, training, and evaluation.

## Dataset Overview
Train Set: 15000 images
Validation Set: 3000 images
Test Set: 3000 images
Classes: 15 different types of vegetables
Image Characteristics:
Size: 224x224 pixels
Color channels: RGB
Default orientation: Images are upright with the top of the image at the top of the viewing area

## Project Objectives
The primary goal of this project is to develop a CNN model capable of accurately classifying various types of vegetables from input images. The project follows these main steps:

### Exploratory Data Analysis (EDA):

Analysis of dataset characteristics.
Visualization of data distributions and image properties.

### Model Development:

Construction of a baseline CNN model with default parameters.
Training and evaluation of the baseline model.

### Model Enhancement:

Development of two additional models:
Model with data augmentation.
Model with data augmentation, batch normalization, and dropout layers.
Training and evaluation of enhanced models.

## Conclusion and Analysis
### EDA Conclusion:

Balanced dataset with 15 classes of vegetables.
Image size: 224x224 pixels, RGB color channels.
Overall images are oriented upright.

### Overall Analysis:

Three models were developed and evaluated.
The final model incorporates batch normalization and dropout layers to address overfitting.
Business insights highlight the transformative benefits of the vegetable recognition model in retail and supply chain operations.

### Strengths and Weaknesses
Strengths:

The final model demonstrates high accuracy and robustness.
Business applications offer substantial efficiency gains and operational excellence.

Weaknesses:

Training process is time and resource-consuming due to the large dataset and multiple model iterations.

## Business Insights
The vegetable recognition CNN model offers transformative benefits in retail and supply chain operations, including retail optimization, streamlined supply chain operations, and enhanced retail experiences. By implementing this model, businesses can achieve increased efficiency, operational excellence, and customer satisfaction.

## Future Directions
Future improvements and directions for the project include:

Regular monitoring and feedback collection for model performance.
Experimentation with different model architectures and augmentation techniques.
Exploration of transfer learning methods.
Updating the model with new datasets or features, such as predicting vegetable quality.
