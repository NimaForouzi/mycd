import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
"""Normalization of images typically involves subtracting the mean and dividing by the standard deviation: normalized_value = original_value − mean / std"""
"""Assume that your original image pixel values are in the range [0, 1], which is common for image data after scaling. Centering Around Zero (Mean = 0.5): Subtracting 0.5 from each pixel
value centers the pixel values around zero. e.g. Original pixel value: 0 becomes 0 − 0.5 = − 0.5 Original pixel value: 0.5 becomes 0.5 − 0.5 = 0 , Original pixel value: 1 becomes 
1 − 0.5 = 0.5. After this step, the pixel values are centered around zero, ranging from -0.5 to 0.5."""

"""Scaling to [-2, 2] (Std = 0.25): Dividing by 0.25 scales the pixel values such that most values fall within the range [-2, 2]. Normalized pixel value for -0.5: − 0.5 / 0.25 = − 2
ormalized pixel value for 0.5: 0.5 0.25 = 2 Therefore, the range of normalized pixel values is now [-2, 2]."""

"""
Problems without Normalization:
Slower Convergence:
Without normalization, the input data may have a large variance, causing the optimization algorithm to converge more slowly.
Gradients can become very large or very small, which might lead to unstable training dynamics.

Difficulty in Training:
Neural networks perform better when inputs are centered and scaled properly. Inputs with varying scales can make the training process more difficult.
The learning algorithm might need a more extended period to adapt to different input scales.

Poor Weight Initialization:
Modern weight initialization methods, such as He or Xavier initialization, assume that the inputs to the network are standardized. Non-normalized inputs can lead to poor initial weights and hinder the training process.
Problems Solved by Normalization

Stable and Faster Training:
Normalized data ensures that the gradients during backpropagation do not explode or vanish, leading to more stable and faster training.
Helps in achieving faster convergence and sometimes better overall performance.

Consistent Input Distribution:
Ensures that each feature contributes equally to the learning process, preventing some features from dominating others.

Generalization:
Networks trained with normalized inputs tend to generalize better, as the network learns to handle a standardized input range effectively.
"""
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

mean = np.array([0.5, 0.5, 0.5]) #pixel_value - mean
std = np.array([0.25, 0.25, 0.25]) #pixel_value - mean / std

data_transforms = {
    'train': transforms.Compose([
        # """(DATA AUGMENTATION) performs a random cropping and resizing operation on an input image. It randomly selects a crop from the image and then resizes this crop to a specified size 
        # (224x224 pixels in this case). This process helps in data augmentation by providing a variety of cropped and resized images, which can improve the generalization of the model."""
        # """(DATA AUGMENTATION) data augmentation technique in PyTorch's torchvision.transforms module that randomly flips an image horizontally with a probability of 0.5. 
        transforms.RandomResizedCrop(224), #224*224 pixel

        # This means that each image has a 50 chance of being flipped left-to-right, and a 50% chance of being left unchanged."""
        transforms.RandomHorizontalFlip(),

        # """
        # transforms.ToTensor() is a transformation provided by the torchvision.transforms module in PyTorch that converts a PIL Image or a NumPy ndarray into a PyTorch tensor. """
        transforms.ToTensor(),
        #normalization
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        # """Resizing the image to a larger size (256x256 pixels) helps retain more details, which can be valuable for capturing fine-grained features during training.
        # Center cropping the resized image to a smaller size (224x224 pixels) allows us to focus on the central part of the image, where the main subject is often located.
        # Therefore, each image undergoes resizing to 256x256 pixels first and then center cropping to 224x224 pixels. This means that all images in the dataset will have a final size of 224x224 pixels after the data transformation process.
        # """(Normalization)resizes the input image to the specified size. The transforms.Resize transformation can take a single integer or a tuple of integers as an argument:
        # Single Integer: Resizes the smaller edge of the image to the given size while maintaining the aspect ratio.Tuple of Integers: Resizes the image to the specified width and height."""
        # resizes the original image to a square of size 256x256 pixels. this transformation does not create a new image
        transforms.Resize(256),
        # """(Normalization)crops the given image at the center to a specified size. In this case, it will crop the image to a 224x224 pixel square.
        #   This helps to standardize the input data for the neural network"""
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #(Normalization)
        transforms.Normalize(mean, std)
    ]),
}

#data folder in home
data_dir = 'data/hymenoptera_data'

# A dictionary with a loop. look at the loop first to know that x can have what values, here [train] and [val] are related values
# s.path.join(data_dir, x) -> data_dir = path | x -> in for loop
image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                          transform=data_transforms[x])
                  for x in ['train', 'val']}
# print(image_datasets['train'])

#Image_Dataset is a dictionary thats why when image_datasets[---] in bracket you should enter 'key' not number of index (like list)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
                                             
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
x, y = dataloaders
#print(x)
#print(dataloaders)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(class_names)


def imshow(inp, title):
    """Imshow for Tensor."""
    #This converts the PyTorch tensor inp to a NumPy array. The numpy() method is a PyTorch function that converts a tensor to a NumPy array
#     rearranges the dimensions of the array. The tuple (1, 2, 0) specifies the new order of dimensions.
# In this case:
# Dimension 1 of the original tensor becomes dimension 0 of the NumPy array. Dimension 2 of the original tensor becomes dimension 1 of the NumPy array. 
#Dimension 0 of the original tensor becomes dimension 2 of the NumPy array. 
#PyTorch Tensor Shape: (channels, height, width)
#numpy shape: ((height, width, channels))
#(height always come before with)
# Purpose:
# This transformation is commonly used when dealing with images in PyTorch. In many image processing libraries (including PyTorch and Matplotlib), the convention for image dimensions is (height, width, channels). However, PyTorch tensors typically have the channel dimension first, followed by height and width. So, transposing the dimensions brings the tensor to the format expected by many image processing libraries, allowing for visualization or further processing.
#1: second(1+1th)  dimension of tensor(width) will be replaced with first (0) dimension of numpy
#2: third(2+1th) dimension of tensor(height) will become second dimension of numpy
#0:first(0+1th) dimension of tensor(channel will become third dimension of numpy)
    inp = inp.numpy().transform(1,2,0)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
# dataloaders againts previous example here is a dictionary, so we should call each key and next-iter for each batch on it
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#an iteratve list
imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)


#### ConvNet as fixed feature extractor ####
# Here, we need to freeze all the network except the final layer.
# We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# every 7 epoch our learning rate multiplied by gamma value

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

