import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import hidden_init


class Network(nn.Module):
    """Abstract class for Actor and Critic networks.

    Agrs:
        state_size (int): Size of the state
        action_size (int): Size of the action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        out_size (int): Size of the output

    """

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units, out_size):
        """Builds the structure of the network"""
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, out_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        raise NotImplementedError


class Actor(Network):
    """Actor network.

    Args:
        state_size (int): Size of the state
        action_size (int): Size of the action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
    """

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model."""
        super(Actor, self).__init__(state_size, action_size, seed, fc1_units, fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(Network):
    """Critic network.

    Args:
        state_size (int): Size of the state
        action_size (int): Size of the action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
    """

    def __init__(self, state_size, action_size, seed, fc1_units, fc2_units):
        """Initialize parameters and build model."""
        super(Critic, self).__init__(state_size, action_size, seed, fc1_units, fc2_units, 1)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)