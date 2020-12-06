import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def main():
    env = gym.make('CartPole-v1')
    pi = Poli