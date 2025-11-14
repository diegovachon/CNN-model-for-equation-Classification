This project implements a Convolutional Neural Network (CNN) for classifying mathematical equations. The model is designed to recognize and classify equations.

Features

Custom CNN architecture with multiple convolutional blocks, advanced data augmentation techniques including mixup, adaptive pooling for handling variable input sizes, dilated convolutions for expanded receptive fields, batch normalization and dropout for regularization, learning rate scheduling with OneCycleLR, comprehensive training pipeline with validation

Model Architecture
The CNN model consists of four main blocks:

First block: 32 filters with 3x3 kernels

Second block: 64 filters with 3x3 kernels

Third block: 128 filters with 3x3 kernels

Fourth block: 256 filters with dilated convolutions

Each block includes:

Convolutional layers, batch normalization, ReLU activation, max pooling

The model uses adaptive pooling and fully connected layers for classification.

Data Augmentation
The project implements several data augmentation techniques: Random affine transformations, Random erasing, Mixup augmentation

Requirements: PyTorch, NumPy, scikit-learn, torchvision

Usage: 
Prepare your dataset in the following format:

Training data: datasets/x_train.npy and datasets/y_train.npy
Test data: datasets/x_test.npy and datasets/y_test.npy

Run the training script:

model, device = train_model()
Evaluate the model:
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

Training Parameters

Batch size: 32

Initial learning rate: 0.001

Number of epochs: 100

Optimizer: AdamW with weight decay

Loss function: Cross Entropy Loss
