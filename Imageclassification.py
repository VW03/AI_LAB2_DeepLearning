import torch
import torchvision
from models import NeuralNetworkClassificationModel
from learners import ClassificationLearner
from utilities import plot_images_and_labels, plot_training_process

## PARAMETERS
random_seed = 1
batch_size_train = 64
batch_size_test = 1000
epochs = 5
learning_rate = 0.01

## INITIALISATION
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

## HARDWARE ACCELERATION
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## DATA
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

#########
"""
training data has shape: (60000, 28, 28)
    - 60000 = number of training images
    - 28 × 28 = pixels per image
    - Each image contains (28 x 28 = 784) input features after flattening
We have 10 targets: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
"""
print("# classes:", torch.unique(train_dataset.targets))
print("train data shape:", tuple(train_dataset.data.shape))
print("train targets shape:", tuple(train_dataset.targets.shape))
print("test data shape:", tuple(test_dataset.data.shape))
print("test targets shape:", tuple(test_dataset.targets.shape))


"""
Activation function tests for NeuralNetworkClassificationModel
---
torch.sigmoid:
train results:	valid loss = 2.2652	valid accuracy = 21.82%
test results:	test loss = 2.2644	test accuracy = 21.91%

torch.relu:
train results:	valid loss = 0.3331	valid accuracy = 90.66%
test results:	test loss = 0.3199	test accuracy = 90.61%

torch.sgn:
train results:	valid loss = 1.1385	valid accuracy = 62.76%
test results:	test loss = 1.1132	test accuracy = 63.71%

torch.tanh:
train results:	valid loss = 0.3388	valid accuracy = 90.46%
test results:	test loss = 0.3295	test accuracy = 90.66%

-> relu because lower test loss and better validation accuracy
"""