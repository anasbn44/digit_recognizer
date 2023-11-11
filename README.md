# Digit Recognizer

## Overview
In this repository, we set the objective to build a Convolutional Neural Network (CNN) using TensorFlow to recognize handwritten digits in the MNIST dataset. To enhance the model's robustness and prevent overfitting, data augmentation, regularization techniques, and dropout will be incorporated.

## Dataset

- **Source**: The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data).
- **Data files** : train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.
- **Data fields** :
Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

## Technologies
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Architecture

- Input Layer : The input layer will be designed to accept 28x28 pixel grayscale images representing handwritten digits.
- Convolutional Layers.
- Pooling Layers: Max pooling layers will be employed to reduce spatial dimensions and retain important features.
- Flatten Layer : The output from the convolutional and pooling layers will be flattened to feed into the fully connected layers.
- Fully Connected Layers :
  - Dense layers will combine extracted features for classification.
  - Regularization techniques such as L2 regularization may be applied to prevent overfitting.
- Dropout : Dropout layers will be added to randomly deactivate neurons during training, improving generalization.
- Data Augmentation : The training data will be augmented using techniques like rotation, width shift, height shift, shear, zoom, and horizontal flip. This will expose the model to various perspectives of the input data, enhancing its ability to generalize.
- Optimization and Loss :
  - The Adam optimizer will be used for efficient weight updates during training.
  - Categorical cross-entropy loss will be employed as it is suitable for multi-class classification tasks.
- Training:
  - The model will be trained on a subset of the MNIST training data.
  - Validation data will be used to monitor model performance and prevent overfitting.
  - Model checkpoints may be saved during training to capture the best-performing model.
- Evaluation:
  - The model's performance will be evaluated on a separate test set to assess its ability to generalize to unseen data.
