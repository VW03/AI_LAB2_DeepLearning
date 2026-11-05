import torch
import torchvision
from models import NeuralNetworkClassificationModel
from learners import ClassificationLearner
from utilities import plot_images_and_labels, plot_training_process
import matplotlib.pyplot as plt

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
train results:	valid loss = 2.2437	valid accuracy = 23.06%
test results:	test loss = 2.2420	test accuracy = 23.35%

torch.relu:
train results:	valid loss = 0.3193	valid accuracy = 91.07%
test results:	test loss = 0.3086	test accuracy = 91.11%

torch.sgn:
train results:	valid loss = 0.6675	valid accuracy = 79.04%
test results:	test loss = 0.6549	test accuracy = 79.73%

torch.tanh:
train results:	valid loss = 0.3282	valid accuracy = 90.67%
test results:	test loss = 0.3184	test accuracy = 90.90%

-> relu because lower loss and better accuracy
"""

#########

"""
# Tests for learning rate and epochs

def run(lr, n_epochs):
    torch.manual_seed(random_seed)                      
    model = NeuralNetworkClassificationModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    learner = ClassificationLearner(model, criterion, optimizer)
    train_losses, valid_losses, valid_accuracies = learner.train(
        train_dataloader, valid_dataloader, n_epochs, device
    )
    return train_losses, valid_losses, valid_accuracies

learning_rates = [0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1]
search_epochs = 5
results = {}

for lr in learning_rates:
    print(f"--- learning rate {lr} ---")
    results[lr] = run(lr, search_epochs)

plt.figure(figsize=(8, 5))
for lr, (tr, va, acc) in results.items():
    plt.plot(range(1, len(va) + 1), va, marker="o", label=f"lr = {lr}")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.yscale("log")
plt.title("Validation loss for different learning rates (batch size 64)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Classification_LR_Search.png", dpi=120)
plt.show()
"""