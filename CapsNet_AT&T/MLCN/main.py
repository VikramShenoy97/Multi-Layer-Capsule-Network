!pip install chart_studio
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from make_graph import plot_graph, generate_reconstructions, generate_encoding_graph
from attMLCN import CapsuleNetwork, CapsuleLoss
import numpy as np
import time
from google.colab import files
from google.colab import drive

drive.mount('/content/drive')
training_folder = "drive/My Drive/ATTFaces/data/faces/training/"
testing_folder = "drive/My Drive/ATTFaces/data/faces/testing/"
device = 'cuda'

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
trainset = datasets.ImageFolder(training_folder, transform=transform)
testset = datasets.ImageFolder(testing_folder, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)

model = CapsuleNetwork()
model.to(device)

optimizer = optim.Adam(model.parameters())
loss_function = CapsuleLoss(reconstruction_alpha=0.0005)
epochs = 800
epoch_interval = [100, 200, 300, 400, 500, 600, 700, 800]
overall_training_accuracy = []
overall_training_loss = []
mode = "Train" # {"Train", "Test", "VPA", "LVG"}

if mode == "Train":
  for epoch in range(400, epochs):
    if((epoch+1) == 500):
      loss_function = CapsuleLoss(reconstruction_alpha=0.0001)
    start_time = time.time()
    running_loss = 0
    training_accuracy = 0
    count = 0
    percent = 0
    for images, raw_labels in iter(trainloader):
      count += 1
      # Generate One Hot encodings of labels
      labels = torch.eye(40).index_select(dim=0, index=raw_labels)
      class_probabilities, reconstructions = model(images.to(device), labels.to(device))
      # Compute Loss and Gradients
      loss = loss_function(images.to(device), reconstructions, labels.to(device), class_probabilities)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      _, logits = class_probabilities.topk(k=1, dim=1)
      # Calculate Accuracy
      training_accuracy_tensor = logits.view(*raw_labels.shape) == raw_labels.to(device)
      training_accuracy += torch.mean(training_accuracy_tensor.type(torch.FloatTensor))
      if(count % (int(len(trainloader)/10)+1) == 0):
        percent += 10
        print("%d %% Complete" % (percent))
    percent += 10
    print("%d %% Complete" % (percent))
    training_loss = running_loss / len(trainloader)
    training_accuracy = training_accuracy * 100 / len(trainloader)
    elapsed_time = time.time() - start_time
    print("Epoch %d / %d:" % (epoch+1, epochs))
    print("Time Elapsed = %d s" % (elapsed_time))
    print("Training Loss = %f" % (training_loss))
    print("Training Accuracy = %0.2f %%" % (training_accuracy))
    if ((epoch+1) % 10 == 0):
      print("Generate Reconstructions:")
      generate_reconstructions(reconstructions.cpu(), images.cpu(), epoch+1)
      print("Reconstructions Generated")
    overall_training_accuracy.append(training_accuracy)
    overall_training_loss.append(training_loss)
    history = {
      "train_loss": overall_training_loss,
      "train_accuracy": overall_training_accuracy
    }
    if((epoch+1) in epoch_interval):
      np.save("history_"+str(epoch+1)+".npy", history)
      torch.save(model, "model_"+str(epoch+1)+".pth")

elif mode == "Test":
  # Testing
  testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=1)
  loaded_model = torch.load("model_400.pth")
  loaded_model.to(device)
  loaded_model.eval()
  with torch.no_grad():
    testing_accuracy = 0
    image_list = []
    reconstructions_list = []
    for images, labels in iter(testloader):
        class_probabilities, reconstructions = loaded_model(images.to(device))
        _, logits = class_probabilities.topk(k=1, dim=1)
        testing_accuracy_tensor = logits.view(*labels.shape) == labels.to(device)
        testing_accuracy += torch.mean(testing_accuracy_tensor.type(torch.FloatTensor))
        image_list.append(images)
        reconstructions_list.append(reconstructions)
        torch.cuda.empty_cache()
  testing_accuracy = testing_accuracy * 100 / len(testloader)
  print("Test Accuracy = %0.2f %%" % (testing_accuracy))
  images_concat = torch.cat(image_list, dim=0)
  reconstructions_concat = torch.cat(reconstructions_list, dim=0)
  generate_reconstructions(reconstructions_concat.cpu(), images_concat.cpu(), "Test")
  plot_graph(testing_accuracy)

elif mode == "VPA":
  # Vector Perturbation Analysis
  loaded_model = torch.load("model.pth")
  loaded_model.eval()
  testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=1)
  # Vector Perturbation Analysis
  for images, _ in iter(testloader):
    _, reconstructions = loaded_model(images.to(device), pose=True, start_val=-0.07)
  generate_reconstructions(reconstructions.cpu())

elif mode == "LVG":
  # Latent Vector Generations
  loaded_model = torch.load("model.pth")
  loaded_model.eval()
  unique_labels = []
  images_ordered = []
  encodings = []
  testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=1)
  # Generate Encodings
  flag=True
  for images, labels in iter(testloader):
    if(labels.item() == 37 and Flag == True):
      test_encoding = loaded_model(images.to(device), encodings=True).reshape(1, -1)
      test_label = labels.item()
      test_image = images.reshape(1, 112, 92)
      flag = False
    elif(labels.item() not in unique_labels):
      encodings.append(loaded_model(images.to(device), encodings=True).unsqueeze(dim=0))
      unique_labels.append(labels.item())
      images_ordered.append(images)
  encodings = torch.cat(encodings, dim=0)
  encodings = encodings.reshape(encodings.shape[0], -1)
  cosine_similarity = nn.CosineSimilarity(dim=-1)

  encoding_list_ordered = []
  for i in range(0, encodings.shape[0]):
    dist = cosine_similarity(test_encoding, encodings[i])
    encoding_list_ordered.append(dist.item())
  tensor_images = torch.cat(images_ordered, dim=0).squeeze()
  generate_encoding_graph(test_image, test_label, tensor_images, encoding_list_ordered, unique_labels)
