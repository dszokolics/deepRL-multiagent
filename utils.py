import torch
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def transpose_list(mylist):
    return list(map(list, zip(*mylist)))


def transpose_to_tensor(input_list):
    return [torch.tensor(element, dtype=torch.float).transpose(0, 1) for element in input_list]
