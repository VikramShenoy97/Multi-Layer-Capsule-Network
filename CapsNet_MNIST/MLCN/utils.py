import torch
import torch.nn as nn
import torch.nn.functional as F

def vector_perturbation(x, y):
  # Shape of x = [11, 10, 16]
  # Shape of y = [11, 10]
  dimension_list = []
  classes = x.shape[1]
  vector_length = x.shape[2]
  # vector_length = 16
  # classes = 10
  for dim in range(0, vector_length):
    alpha = -0.25
    interval_list = []
    while(alpha <= 0.25):
      temp = x.contiguous().view(1, -1, vector_length).clone()
      # Shape of temp = [1, 110, 16]
      # Add perturbation value alpha to each dimension
      temp[0, :, dim] = temp[0, :, dim] + alpha
      alpha = alpha + 0.05
      interval_list.append(temp)
    temp = torch.cat(interval_list, dim=0).unsqueeze(dim=0)
    # Shape of temp = [1, 11, 110, 16]
    dimension_list.append(temp)
  x = torch.cat(dimension_list, dim=0)
  # Shape of x = [16, 11, 110, 16]
  repeat_y = x.shape[0] * x.shape[1]
  # repeat_y = 176
  x = x.view(-1, classes, vector_length)
  # Shape of x = [1936, 10, 16]
  y = y.unsqueeze(dim=0)
  # Shape of y = [1, 11, 10]
  y_list = []
  for _ in range(repeat_y):
    y_list.append(y)
  y = torch.cat(y_list, dim=0)
  # Shape of y = [176, 11, 10]
  y = y.view(-1, y.shape[2])
  # Shape of y = [1936, 10]
  return x, y

def squash(tensor, dim=-1):
  # Square of Absolute Value
  squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
  return (squared_norm / (1 + squared_norm)) * (tensor / torch.sqrt(squared_norm))

def softmax(tensor, dim=-1):
  # Shape of tensor = [10 x 128 x 1152 x 1 x 16]
  transposed_tensor = tensor.transpose(dim, len(tensor.shape)-1)
  # Shape of tensor = [10 x 128 x 16 x 1 x 1152]
  softmaxed_tensor = F.softmax(transposed_tensor.contiguous().view(-1, transposed_tensor.shape[-1]), dim=-1)
  # Shape of softmaxed_tensor = [20480 x 1152]
  output_tensor = softmaxed_tensor.view(*transposed_tensor.shape).transpose(dim, len(tensor.shape)-1)
  # Shape of output_tensor = [10 x 128 x 1152 x 1 x 16]
  return output_tensor
