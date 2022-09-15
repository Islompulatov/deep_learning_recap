import torch
import torch.nn.functional as F
from torch import nn

from collections import OrderedDict

# YOUR CODE HERE

def neural_net(input_size, hidden_layer1, hidden_layer2, out_size):

    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_layer1)),
        ('act1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
        ('act2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_layer2, out_size))
        ]))

    return model    




        