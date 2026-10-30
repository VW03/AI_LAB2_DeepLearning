import torch


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the linear regression layer.
        self.linear_layer = torch.nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor):
        # Implement the forward pass.
        return self.linear_layer(x)


class NeuralNetworkClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # the three classification layers:
        # each image contains (28 x 28 = 784) pixels per image -> input features = 784
        self.layer1 = torch.nn.Linear(784, 392)
        self.layer2 = torch.nn.Linear(392, 196)
        self.layer3 = torch.nn.Linear(196, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x
