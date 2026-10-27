import os
import math
import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader


class Learner():

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def save(self, label):
        if not os.path.exists(os.path.join("checkpoints")):
            os.makedirs(os.path.join("checkpoints"))
        torch.save(self.model.state_dict(), os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label))

    def load(self, label):
        self.model.load_state_dict(torch.load(os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label), weights_only=True))
    

class RegressionLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []

        for epoch in range(epochs):

            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in dataloader: #batch size staat in run_linear_regeression op 32 dus 32 records/keer
                data = data.to(device)
                targets = targets.to(device)

                # Implement the batch update step.

                # De 5 vaste stappen van elke PyTorch-trainingsstap
                # (dit is de automatische versie van wat we in deel I manueel deden)

                # 1. Gradients van de vorige batch wissen.
                #    PyTorch TELT gradients standaard OP in .grad. Zonder dit
                #    zou elke batch de gradients van alle vorige batches erbij
                #    krijgen en zou het model in de verkeerde richting stappen.
                self.optimizer.zero_grad()

                # 2. Forward pass: voorspelling van het model voor deze batch.
                #    (= calculate_output in deel I)
                outputs = self.model(data)

                # 3. Loss berekenen: verschil tussen voorspelling en target
                #    met het criterion dat in run_linear_regression.py gekozen is.
                #    (= calculate_loss in deel I)
                loss = self.criterion(outputs, targets)

                # 4. Backward pass: autograd berekent d(loss)/d(parameter) voor
                #    elke parameter en zet ze in parameter.grad.
                #    (= calculate_gradients in deel I, nu automatisch)
                loss.backward()

                # 5. De optimizer past de parameters aan met die gradients
                #    (bij SGD: w = w - learning_rate * gradient).
                #    (= update_weights in deel I)
                self.optimizer.step()

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(dataloader.dataset)

            train_losses.append(train_loss)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.6f}")   

            if train_loss <= min(train_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return train_losses

    @torch.no_grad()
    def test(self, dataset: Dataset, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        test_loss = 0

        # Compute the model's RMSE on the test dataset.
        dataloader = DataLoader(dataset, batch_size=32)

        sum_squared_error = 0.0

        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            outputs = self.model(data)

            sum_squared_error += torch.nn.functional.mse_loss(outputs, targets, reduction="sum").item()

        mse = sum_squared_error / len(dataset)
        test_loss = math.sqrt(mse)

        return test_loss

    @torch.no_grad()
    def predict(self, dataset, device):
        self.model.to(device)
        self.model.eval()
        data = dataset[:][0]
        data = data.to(device)
        results = self.model(data)
        return results


class ClassificationLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(epochs):
            
            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in train_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                
                # TODO: Implement the batch update step.

                loss = torch.tensor([0])

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_dataloader.dataset)

            self.model.eval()  # set model in evaluation mode

            # evaluate the model on the validation dataset
            valid_loss, valid_accuracy = self._evaluate(valid_dataloader, device)    

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.4f}\tvalid loss = {valid_loss:.4f}\tvalid accuracy = {100 * valid_accuracy:.2f}%")   

            if valid_loss <= min(valid_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return (train_losses, valid_losses, valid_accuracies)

    def test(self, test_dataloader: DataLoader, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        # evaluate the model on the test dataset
        test_loss, test_accuracy = self._evaluate(test_dataloader, device)

        return (test_loss, test_accuracy)

    @torch.no_grad()
    def _evaluate(self, dataloader, device):
        eval_loss = 0
        eval_accuracy = 0

        # TODO: Compute the model's loss and accuracy over the dataset.

        return (eval_loss, eval_accuracy)
