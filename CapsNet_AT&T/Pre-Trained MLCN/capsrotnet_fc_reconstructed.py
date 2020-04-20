import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import squash, softmax, vector_perturbation

device = 'cuda'

class CapsLayerOne(nn.Module):
  def __init__(self, num_capsules, in_channels, out_channels, kernel_size, stride):
    super(CapsLayerOne, self).__init__()
    # Repeat Convolutions based on number of capsules (64 in this case).
    self.capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in range(num_capsules)])

  def forward(self, x):
    # Shape of x = [2 x 256 x 52 x 52]
    # Apply Convolutions to Input x to generate capsules.
    outputs = [capsule(x) for capsule in self.capsules]
    # Shape of each output in list = [2 x 8 x 22 x 22 x 1]
    outputs = torch.cat(outputs, dim=-1)
    # Shape of outputs = [2 x 8 x 22 x 22 x 64]
    outputs = outputs.view(outputs.shape[0], outputs.shape[1], -1)
    # Shape of outputs = [2 x 8 x 30976]
    outputs = outputs.transpose(1, len(outputs.shape)-1)
    # Shape of outputs = [2 x 30976 x 8]
    outputs = squash(outputs)
    # Shape of outputs = [2 x 30976 x 8]
    return outputs

class IntermediateCapsLayer(nn.Module):
  def __init__(self, num_capsules, routing_nodes, in_channels, out_channels, routing_iterations=3):
    super(IntermediateCapsLayer, self).__init__()
    self.routing_iterations = routing_iterations
    # Weights = [40 x 7744 x 8 x 32]
    self.weights = torch.randn(num_capsules, routing_nodes, in_channels, out_channels).to(device)

  def forward(self, x):
    # Shape of x = [2 x 7744 x 8]
    # Shape of x = [1 x 2 x 7744 x 1 x 8], Shape of Weights = [40 x 1 x 7744 x 8 x 32]
    x_hat = torch.matmul(x[None, :, :, None, :], self.weights[:, None, :, :, :])
    # Shape of x_hat = [40 x 2 x 7744 x 1 x 32]
    # b is a temporary variable that will store the value of routing weights c and will be gradually updated.
    b = torch.zeros(*x_hat.shape).to(device)
    for i in range(self.routing_iterations):
      # Routing weights for all capsules of layer l (i.e dim_2 = 23936)
      c = softmax(b, dim=2)
      # Weighted sum of x_hat and routing weights c across all capsules of layer l (i.e. Sum over dim_2 = 23936)
      outputs = squash((x_hat*c).sum(dim=2, keepdim=True))
      # Shape of outputs = [40 x 2 x 1 x 1 x 32]
      if(i != self.routing_iterations-1):
        # Weight Update Step: Update weight b using dot product similarity.
        db = (x_hat * outputs).sum(dim=-1, keepdim=True)
        # Shape of db = [40 x 2 x 23936 x 1 x 1]
        b = b + db
    return outputs


class CapsuleNetwork(nn.Module):
  def __init__(self):
    super(CapsuleNetwork, self).__init__()
    self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=2, padding=0)
    self.capslayerone = CapsLayerOne(num_capsules=64, in_channels=256, out_channels=8, kernel_size=9, stride=4)
    self.capslayertwo = IntermediateCapsLayer(num_capsules=48, routing_nodes=64*11*11, in_channels=8, out_channels=24)
    self.capslayerthree = IntermediateCapsLayer(num_capsules=40, routing_nodes=48, in_channels=24, out_channels=32)
    self.fc = nn.Sequential(nn.Linear(32*40, 100), nn.ReLU(inplace=True), nn.Linear(100, 4))
    self.decoder = nn.Sequential(nn.Linear(4, 100), nn.ReLU(inplace=True), nn.Linear(100, 32*40), nn.ReLU(inplace=True), nn.Linear(32*40, 2048), nn.ReLU(inplace=True), nn.Linear(2048, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 12544), nn.Sigmoid())
  def forward(self, x, y=None, pose=False, encodings=False, start_val=-0.05):
    # Encoder Network
    # Layer 1: Convolutional Layer
    # Input = [2 x 1 x 112 x 112]
    # Output = [2 x 256 x 52 x 52]
    x = F.relu(self.conv(x), inplace=True)
    # Layer 2: Capsule Layer One
    # Input = [2 x 256 x 52 x 52]
    # Output = [2 x (64*22*22) x 8] = [1 x 30976 x 8]
    x = self.capslayerone(x)
    # Layer 3: Capsule Layer Two
    # Input = [2 x 30976 x 8]
    # Output = [50 x 2 x 1 x 1 x 24]
    x = self.capslayertwo(x)
    x = x.transpose(0, 1).squeeze()
    if(len(x.shape) == 2):
        x = x.unsqueeze(dim=0)
    # Shape of x = [2, 50, 24]
    if(encodings is True):
        return x.reshape(x.shape[0], -1)
    # Layer 4: Capsule Layer Three
    # Input = [2 x 50 x 24]
    # Output = [40 x 2 x 1 x 1 x 32]
    x = self.capslayerthree(x)
    x = x.transpose(0, 1).squeeze()
    # if(len(x.shape) == 2):
    #     x = x.unsqueeze(dim=0)
    # # Shape of x = [2, 40, 32]
    # if(encodings is True):
    #     return x.reshape(x.shape[0], -1)
    # # Layer 4: Capsule Layer Three
    # # Shape of x = [2, 40, 32]
    # # Output = [4 x 2 x 1 x 1 x 10]
    # x = self.capslayerfour(x)
    # x = x.transpose(0, 1).squeeze()
    # if(len(x.shape) == 2):
    #     x = x.unsqueeze(dim=0)
    # Shape of x = [2, 4, 10]
    class_scores = torch.sqrt((x**2).sum(dim=-1))
    # Shape of class_scores = [2 x 40]
    class_probabilities = F.softmax(class_scores)
    # Shape of class_probabilities = [2 x 40]
    if y is None: # During Testing
      _, labels = class_probabilities.topk(k=1, dim=1)
      # Generate one hot encoded labels from class probabilities
      y = torch.eye(40).to(device).index_select(dim=0, index=labels.squeeze())
    if pose == True:
      # During Vector Perturbation Analysis
      x, y = vector_perturbation(x, y, start_val)
    # Decoder Network
    # Layer 5: Fully Connected Layer 1
    # Input = 32*40 = 1280 (x[2 x 40 x 32] * y[2 x 40 x 1](one-hot encoded labels) is done to ignore incorrect vectors through 0 masking)
    # Output = 2048
    # Layer 6: Fully Connected Layer 2
    # Input = 2048
    # Output = 4096
    # Layer 7: Fully Connected Layer 3
    # Input = 4096
    # Output = 112*92 = 10304
    x = x.reshape(x.shape[0], -1)
    probs = self.fc(x)
    reconstructions = self.decoder(probs)
    return probs, reconstructions


class CapsuleLoss(nn.Module):
  def __init__(self, reconstruction_alpha=0.0005):
    super(CapsuleLoss, self).__init__()
    self.reconstruction_loss = nn.MSELoss(size_average=False)
    self.m_positive = 0.9
    self.m_negative = 0.1
    self.lambda_constant = 0.5
    self.alpha_constant = reconstruction_alpha

  def forward(self, images, reconstructions, labels, class_probabilities):
    assert torch.numel(images) == torch.numel(reconstructions)
    images = images.view(reconstructions.shape[0], -1)
    # Compute Margin Loss (ReLU performs the same functionality as max)
    margin_loss = (labels * F.relu(self.m_positive - class_probabilities, inplace=True)**2 + self.lambda_constant * (1. - labels) * F.relu(class_probabilities - self.m_negative) ** 2).sum()
    # Compute Reconstruction Loss
    reconstruction_loss = self.reconstruction_loss(images, reconstructions)
    # Return Overall Loss
    return (margin_loss + self.alpha_constant*reconstruction_loss)/images.shape[0]
