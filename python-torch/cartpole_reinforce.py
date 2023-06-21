#!/usr/bin/env python3

from cartpole import CartPoleEnv
from simplecartpolepolicy import SimpleCartPolePolicy
import torch.optim as optim
from device import get_device
from reinforce import reinforce
import numpy
numpy.random.seed(42)


device = get_device()
hyperparameters = {
    "n_training_episodes": 500,
    "n_evaluation_episodes": 500,
    "max_t": 1000,
    "gamma": 0.99,
    "lr": 1e-2,
}
policy = SimpleCartPolePolicy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=hyperparameters["lr"])
env = CartPoleEnv()
reinforce(policy=policy,
          env=env,
          gamma=hyperparameters['gamma'],
          optimizer=optimizer,
          n_training_episodes=hyperparameters['n_training_episodes'],
          batch_size=16,
          device=device)
