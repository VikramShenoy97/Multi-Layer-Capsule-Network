import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class decoder(nn.Module):

    def __init__(self):

        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module("linear 1", nn.Linear(160, 512))
        self.model.add_module("relu 1", nn.ReLU())
        self.model.add_module("linear 2", nn.Linear(512, 1024))
        self.model.add_module("relu 2", nn.ReLU())
        self.model.add_module("linear 3", nn.Linear(1024, 784))
        self.model.add_module("relu 3", nn.Sigmoid())

        self.optimizer = optim.Adam(self.model.parameters())

    def forward(self, digicaps):

        return self.model.forward(digicaps)

    def train(self, capsule_outputs, targets):

        for i, capsule_output in enumerate(capsule_outputs):

            self.optimizer.zero_grad()
            output = self.forward(capsule_output)
            loss = F.mse_loss(output, targets[i])
            loss.backward()
            self.optimizer.step()

decoder = decoder()

capsule_outputs = torch.randn(5, 160)
targets = torch.randn(5, 784)

decoder.train(capsule_outputs, targets)
