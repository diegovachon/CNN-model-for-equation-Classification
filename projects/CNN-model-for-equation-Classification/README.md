# CNN Model for Equation Classification

This project implements a Convolutional Neural Network (CNN) for classifying mathematical equations. The model is designed to recognize and classify equations, achieving high accuracy through advanced deep learning techniques.

## Features

- Custom CNN architecture with multiple convolutional blocks
- Advanced data augmentation techniques including mixup
- Adaptive pooling for handling variable input sizes
- Dilated convolutions for expanded receptive fields
- Batch normalization and dropout for regularization
- Learning rate scheduling with OneCycleLR
- Comprehensive training pipeline with validation

## Model Architecture

The CNN model consists of four main blocks:

1. First block: 32 filters with 3x3 kernels
2. Second block: 64 filters with 3x3 kernels
3. Third block: 128 filters with 3x3 kernels
4. Fourth block: 256 filters with dilated convolutions

Each block includes:
- Convolutional layers
- Batch normalization
- ReLU activation
- Max pooling

The model uses adaptive pooling and fully connected layers for classification.

## Data Augmentation

The project implements several data augmentation techniques:
- Random affine transformations
- Random erasing
- Mixup augmentation

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- torchvision

## Usage

1. Prepare your dataset in the following format:
   - Training data: `datasets/x_train.npy` and `datasets/y_train.npy`
   - Test data: `datasets/x_test.npy` and `datasets/y_test.npy`

2. Run the training script:
```python
model, device = train_model()
```

3. Evaluate the model:
```python
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
```

## Training Parameters

- Batch size: 32
- Initial learning rate: 0.001
- Number of epochs: 100
- Optimizer: AdamW with weight decay
- Loss function: Cross Entropy Loss

## Model Performance

The model achieves high accuracy through:
- Advanced architecture design
- Comprehensive data augmentation
- Learning rate scheduling
- Regularization techniques

## Author

Diego Vachon Galindo

## Acknowledgments

- PyTorch team for the deep learning framework
- Contributors to the torchvision library 