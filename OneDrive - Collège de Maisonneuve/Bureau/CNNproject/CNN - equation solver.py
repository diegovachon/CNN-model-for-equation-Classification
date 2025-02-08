import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import pandas as pd


"""
This CNN model has a design intended to extract and classify features from grayscale images representing equations
"""
class CNN(nn.Module):
    def __init__(self, num_classes=17):
        super(CNN, self).__init__()

        """The pattern above repeats in blocks (second, third, and fourth), with each block increasing the number of filters (channels) and applying
        downsampling through max pooling. This setup progressively extracts more complex features while reducing spatial dimensions.
        The fourth block uses dilated convolutions, which expand the receptive field without increasing the number of parameters. """

        self.features = nn.Sequential(
            # It defines the first layer group
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Convolution layer with 1 input channel (grayscale image), 32 filters, and a 3x3 kernel
            nn.BatchNorm2d(32),  # Batch normalization layer for stabilizing and speeding up training
            nn.ReLU(inplace=True),   # Activation function to introduce non-linearity
            nn.Conv2d(32, 32, kernel_size=3, padding=1),    # We repeat the process for better representation
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),      # Downsampling operation that reduces spatial dimensions by half (pooling layer)

            # Second block  (repeats a similar pattern, increasing the number of filters)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third block   (repeats a similar pattern, increasing the number of filters)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Fourth block with dilated convolutions
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        """ Layer ensures the output has a fixed size of 4x4 spatial dimensions, regardless of the input size
        which allows for more flexibility with varying input image sizes """
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # The model flattens the output and feeds it through a series of fully connected layers for classification
        self.classifier = nn.Sequential(
            # It takes the flattened features and connects them to a dense layer with 1024 neurons. This high number of neurons captures complex interactions among the features
            nn.Linear(256 * 4 * 4, 1024),

            nn.ReLU(inplace=True),          # Activation function
            nn.Dropout(0.4),                # It prevents overfitting by randomly dropping out neurons during training
            nn.Linear(1024, 512),  # reduces the number of neurons to 512, preparing for a more compact representation of the features
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes) # maps the 512 features down to the number of output classes (17), corresponding to the possible results of the equations
        )
    '''
    This function does the following : Convolutional and pooling operations in self.features, adaptive average pooling to a fixed size
    and flattening and passing through fully connected layers in self.classifier
    '''
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # It reshapes the feature map into a 1D vector, preparing it for the fully connected layers
        x = self.classifier(x)
        return x



"""
get_transforms() purpose : It does a data augmentation that is used to artificially increase the size and the variability of a training dataset.
It just applies random transformations to images to help the model learn more robust features and to improve its generalization to unseen data.
"""
def get_transforms():
    """ It creates a pipeline of image transformations using transforms.Compose from torchvision.transforms
    which only applies each transformation sequentially """
    train_transforms = transforms.Compose([

        transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),

        #RandomErasing() purpose : It randomly erases small rectangular regions of the image with a given probability. It will encourages the model
        #to focus on different parts of the image by obscuring certain regions; p = 0.2 is the probability of applying this transformation to each
        #image is 20%; the parameter about scale it helps to specify the range of the proportion that can be erased. In this case, between 2% and 10%
        #can be randomly erased, making these erased areas small. So, it helps the model to generalize by learning features from incomplete images.

        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    return train_transforms




# Training function with mixup :  This function defines a single epoch of the training process, which involves running through all batches in the train_loader
def train_one_epoch(model, train_loader, criterion, optimizer, device, transforms=None):
    model.train()
    running_loss = 0.0   # Initializes a variable to accumulate the loss over each batch in this epoch, which will later be averaged and returned

    for inputs, labels in train_loader: # Iterates over each batch in the training data
        # Moves the batch of inputs and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)  # inputs -> images, labels -> corresponding target labels for each batch

        # Apply transforms if there is one available
        if transforms:
            inputs = transforms(inputs)

        """ Mixup augmentation. Why mixup augmentation? To improve mainly model generalization (is useful to prevent overfitting),
        robustness (reduces the model's sensitivity to small changes in the input) and it can lead to reduced memorization
        because Mixup reduces the likelihood of memorizing individual samples by combining them. """

        if np.random.random() < 0.5:  # Determines whether to apply the mixup augmentation based on a 50% probability
            """ Generates a mixing factor lam for mixup using Beta distribution
            Beta distribution : It provides values between 0 and 1, making it useful for mixing data in weighted proportions
            """
            lam = np.random.beta(0.2, 0.2)
            # Creates a random permutation of indices for the batch; shuffle the batch so that each sample is paired with a randomly selected other sample for mixup
            idx = torch.randperm(inputs.size(0))
            # Creates mixed inputs by combining each image in the batch with another randomly selected image, weighted by lam
            mixed_inputs = lam * inputs + (1 - lam) * inputs[idx]
            """ Calculates the loss for the mixed inputs using the original and shuffled labels; outputs : model generates predictions for the mixed
            inputs; criterion(): calculates the loss between the output and the original labels; criterion(outputs, labels[idx]) : Calculates the loss
            between the output and the shuffled labels.
            """
            outputs = model(mixed_inputs)
            # The final loss is a weighted combination of the two losses, with weights lam and 1 - lam.
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[idx])
        else:
            #  Standard forward pass and loss calculation when mixup is not applied
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()  # Clears old gradients to prevent accumulation from previous batches
        loss.backward()  # Computes the gradients of the loss with respect to the model parameters
        # Gradient clipping : Clipping ensures that the gradient norm does not exceed max_norm=1.0, which can stabilize training and help the model converge
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Updates model parameters based on the gradient
        optimizer.step()

        # Adds the batch loss to running_loss to track the cumulative loss over all batches
        running_loss += loss.item()
    # Returns avg loss
    return running_loss / len(train_loader)




# Load and preprocess data
x_train = np.load('x_train.npy') / 255.0   # Division by 255 normalizes the image data to a range [0,1], generally beneficial to NN training
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy') / 255.0
y_test = np.load('y_test.npy')

# Convert to PyTorch tensors (necessary for training in PyTorch) and it adds the required channel dimension for grayscale images
x_train = torch.tensor(x_train).float().unsqueeze(1)   # .unsqueeze(1) adds a channel dimension (for grayscale images)
y_train = torch.tensor(y_train).long()   # .long to ensure the target labels are in integer format (required for CrossEntropyLoss)
x_test = torch.tensor(x_test).float().unsqueeze(1)
y_test = torch.tensor(y_test).long()




# train_model() function purpose : Does the process of setting up, training and validating the model
def train_model():
    """ Device configuration : it looks whether GPU is available for computation. If GPU is available, it assigns the computations to the GPU to
    speed up training and otherwise, it goes to the CPU. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CHECK TO REMOVE THIS LINE

    # Model initialization : It initializes the CNN model and sends it to the device (CPU or GPU) specified. It prepares the model for training
    model = CNN().to(device)

    """ Hyperparameters. Why this choice of hyperparameters? I chose a batch size of 32 because it's a good balance between computational efficiency
    and the model performance. I tried with 64 and 128 as the batch size, but the results were less accurate (even if larger batch size increase speed
    but it decreases model generalization a bit). Also, I tried with 16 but the model was too slow for not that much increase in accuracy. Then, I also
    chose an initial learning rate of 0.001. Initially, I tried a smaller learning rate since it allows the model to converge slowly (which helps to
    find the optimal solution), but the time it takes to converge was way too slow. I found a great balance with a lr of 0.001 as I did find what I
    think is the optimal solution. Finally, I chose 100 epops because it the correct number of epops since the model is able to converge to the optimal
    solution within 100 epops. I tried with more than 100 epops and it leads to overfitting (and it takes more time). Less than 100 epops does not let
    the model to converge to the optimal solution. So, 100 epops is a perfect balance to converge to the optimal solution.
    """

    batch_size = 32
    initial_lr = 0.001
    num_epochs = 100

    """ Split data """
    # It combines x_train and y_train into a single TensorDataset which allows to easy access to paired data during training
    train_dataset = TensorDataset(x_train, y_train)
    # Calculates the sizes for training and validation sets, using 90% for training and 10% for validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    # Splits train_dataset into train_dataset and val_dataset using random_split, to enable validation during training
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    """ Data loaders : It creates data loaders for the training, validation and test sets (since data loaders are an efficient way to iterate over
    data in batches during training and evaluation.
    shuffle = True does shuffle the data at each epoch to prevent the model to learn the order of data fpr train_loader. For validation and test
    loaders, shuffle is set to False to keep data order consistent. num_workets = 4 enables parallel data loading to speed up data preparation
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(TensorDataset(x_test[:4500], y_test), batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    # It defines the loss function that will measure how well the model's predictions match the true label (used for multi-class classification task)
    criterion = nn.CrossEntropyLoss()
    """ It defines the optimizer, which updates the model parameters to minimize the loss function. I used here AdamW, a variant of the Adam optimizer,
    which includes weight decay (L2 regularization) which helps to prevent overfitting (it penalizes large weights)
    model.parameters : tells optimizer which parameters to update; lr is set to initial_lr, which has been defined earlier; weigh_decay adds a
    penalty to large weights to improve generalization of the model
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    # Learning rate scheduler purpose : It dynamically adjusts the learning rate during training to improve convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,   # Use to adjust the learning rate
        max_lr=initial_lr,  # Sets the maximum learning rate
        epochs=num_epochs,  # Specifies the total number of epochs for the entire cycle
        steps_per_epoch=len(train_loader), # Calculates how many steps (batches) there are in each epoch, allowing the scheduler to adjust the rate for each batch
        pct_start=0.1,   # Specifies the percentage of the cycle during which the learning rate will increase (10% in this case)
        anneal_strategy='cos'  # It uses a cosine annealing strategy, which gradually reduces the learning rate following a cosine curve
    )

    # Training loop
    best_val_acc = 0.0  # Searching for the best validation accuracy
    transforms = get_transforms()    # Applies data augmentation transforms to increase the model's robustness
    for epoch in range(num_epochs):  # Iterates through the specified number of training epochs (i.e. 100)
        # Trains the model for a single epoch; train_loss stores the average loss over all batches for that epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, transforms)

        # Sets the model to evaluation mode and prepares to validate it on the validation set
        model.eval()
        val_preds, val_labels = [], []    # Stores the predicted and actual labels
        with torch.no_grad():  # It disables gradient computation (it speeds up computation since not updating weights during validation)
            for inputs, labels in val_loader:  # Iterates through the validation data and collects predictions
                inputs, labels = inputs.to(device), labels.to(device)   #  transfer data to the chosen device CPU or GPU
                outputs = model(inputs)   # makes predictions for the current batch of inputs
                _, preds = torch.max(outputs, 1)   # retrieves the class with the highest probability for each input
                val_preds.extend(preds.cpu().numpy())  # Convert to NumPy array and stores predicted labels
                val_labels.extend(labels.cpu().numpy())   # Stores the actual labels and conversion to NumPy

        # Calculates and displays the validation accuracy for the current epoch
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        scheduler.step()   # Updates the learning rate based on the learning rate scheduler lead to better
                           # performance by optimizing learning rates throughout training

    return model, device    # It returns the trained model and the device used for the training


model, device = train_model()
# Loads the best model’s weights from a saved file into the current model instance and evaluate it
#model.load_state_dict(torch.load("best_model2.pth"), weights_only = True)
model.eval()
# Create data loader for evaluation (first 4500 samples)
test_loader = DataLoader(
    TensorDataset(x_test[:4500], y_test),  # Only use the first 4500 samples for evaluation
    batch_size=64,
    shuffle=False
)

# Create data loader for Kaggle submission (all 9000 samples)
test_loader_full = DataLoader(
    x_test,  # Use all 9000 samples without labels
    batch_size=64,
    shuffle=False
)

# Evaluate on first 4500 samples
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Calculate accuracy
test_accuracy = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy (first 4500 samples): {test_accuracy * 100:.2f}%")


# Generate predictions for the entire test set (9000 samples)
full_predictions = []

with torch.no_grad():
    for inputs in test_loader_full:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        full_predictions.extend(preds.cpu().numpy())


# Convert to NumPy array
full_predictions = np.array(full_predictions)
# Save predictions to CSV
def generate_csv_kaggle(y):
    indexes = np.arange(len(y))
    csv_labels = np.concatenate((indexes.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(csv_labels, columns=["Id", "Category"])
    df.to_csv("predicted_labels.csv", index=False)
    print("Submission file 'predicted_labels.csv' has been created!")

generate_csv_kaggle(full_predictions)