import torch
from torchvision import datasets, transforms
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from make_graph import plot_graph, generate_reconstructions, generate_encoding_graph
from attMLCN import CapsuleNetwork, CapsuleLoss, CapsLayerOne
from capsrotnet import CapsuleNetwork as cn
from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
import pandas as pd
import time
import seaborn as sns




class TempNetwork(nn.Module):
  def __init__(self):
    super(TempNetwork, self).__init__()
    self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=2, padding=0)
    self.capslayerone = CapsLayerOne(num_capsules=64, in_channels=256, out_channels=8, kernel_size=9, stride=4)

  def forward(self, x, y=None, conv=False):
    if conv:
        return self.conv(x)
    x = F.relu(self.conv(x), inplace=True)
    _, feature_maps = self.capslayerone(x)
    return feature_maps


def get_mean_filter(capsule, visualization=False):

    if visualization:
        weights = capsule.data.numpy()
    else:
        weights = capsule.weight.data.numpy()

    weights = np.mean(weights, axis=0)
    weights = np.mean(weights, axis=0)

    return weights


def get_max_filter(capsule, visualization=False):

    if visualization:
        weights = capsule.data.numpy()
    else:
        weights = capsule.weight.data.numpy()

    max = 0
    idx = 0

    weights = np.squeeze(weights)

    for i, filters in enumerate(weights):

        if np.sum(filters) > max:

            max = np.sum(filters)
            idx = i

    return weights[i]




def generate_heatmap(filter):
    n = filter.shape[0]
    X1 = np.linspace(0, n-1, n)
    X2 = np.linspace(n-1, 0, n)
    X = np.zeros((n**2, 3))

    x1, x2 = np.meshgrid(X1, X2)

    X[:, 0] = x1.flatten()
    X[:, 1] = x2.flatten()

    filter = filter.flatten()

    for i in range(n**2-1, -1, -1):
        X[i, 2] = filter[i]

    X = pd.DataFrame(X, columns=['X1', 'X2', 'Y'])
    X = X.round(4)
    X = pd.pivot_table(X, values='Y',
                         index='X2',
                         columns='X1')

    ax = sns.heatmap(X, cmap='gray')
    ax.invert_yaxis()

    plt.show()







def visualize_feature_maps(filepath, filename, image, idx):

    device = 'cpu'

    model = torch.load(filepath)

    model.to(device)

    temp_model = TempNetwork()

    temp_model.conv = model.conv
    temp_model.capslayerone.capsules = model.capslayerone.capsules

    feature_maps = np.squeeze(temp_model.forward(image, conv=True).data.numpy())

    rows = 4
    columns = 4
    fig, axs = plt.subplots(rows, columns)
    count = 0
    for i in range(0, rows):
        for j in range(0, columns):
                  axs[i,j].imshow(feature_maps[idx[count]], cmap="gray")
                  axs[i,j].axis("off")
                  count += 1

    fig.savefig(filename)
    # plt.show()



training_folder = "data_mlcn/faces/temp/"
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
trainset = datasets.ImageFolder(training_folder, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
count = 0
for images, raw_labels in iter(trainloader):
    if count == 0:
        temp_image1 = images
    else:
        temp_image2 = images
    count += 1




idx = np.random.permutation(256)

visualize_feature_maps("Trained Models/Pretraining/Model_100_FC_Reconstructed/model_800.pth", "Model_100_FC_Reconstructed.png", temp_image1, idx)

visualize_feature_maps("Trained Models/Baseline/model_800.pth", "Baseline.png", temp_image2, idx)
