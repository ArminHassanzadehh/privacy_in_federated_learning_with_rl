# Privacy Control in Federated Learning via Reinforcement Learning

## Problem Statement
The central problem addressed in this project is:
How can privacy mechanisms in federated learning be adapted dynamically during training to better balance model performance and privacy leakage?

## Proposed Approach

In this work, privacy control is formulated as a sequential decision-making problem, where the goal is to dynamically adjust the level of noise applied to gradients during federated training.

A reinforcement learning (RL) agent is introduced as a controller that:
- observes the current state of the system
- selects actions that modify the privacy parameter (noise level)
- receives feedback based on both model performance and privacy leakage

### Components of the System

- #### Federated Learning Simulation
  A simplified multi-client setup where each client trains a local model on a subset of data and shares updates with a central aggregator.
- #### Privacy Mechanism
  Noise is injected into gradients during local training to simulate differential privacy behavior.
- #### RL Controller
  A policy-gradient-based agent (PPO) adjusts the noise level during training based on observed system behavior.

### State Representation

The agent observes:
- model accuracy (utility signal)
- current noise level

### Action Space

- increase noise
- decrease noise
- keep noise unchanged

### Reward Function

The reward balances utility and privacy:

Reward = Accuracy − λ × PrivacyLeakage

where privacy leakage is approximated using model confidence as a proxy.


## Installation
You can install the required packages with:
```bash
pip install torch torchvision numpy matplotlib gym stable-baselines3
