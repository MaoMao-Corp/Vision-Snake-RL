# Screen Capture RL Snake Player

This project uses computer vision techniques to capture and process the game screen of a Snake game, making it possible to apply reinforcement learning (RL) to control the snake.

The objective of the project was to make an agent agnostic to browser, resolution, window size and game skin that runs on virtually any hardware and browser.

## Table of Contents

- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Screen Capture Functionality](#screen-capture-functionality)
- [Preprocessing](#preprocessing)
- [Hough Line Detection](#hough-line-detection)
- [Grid Detection and Cell Saving](#grid-detection-and-cell-saving)
- [Reinforcement Learning Architecture](#reinforcement-learning-architecture)
- [Training Methodology](#training-methodology)
- [Run the Script](#run-the-script)

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3D24?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)
![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2D9E64?style=flat&logo=stable-baselines3&logoColor=white)

## General Overview

The screen capture part of the system extracts important visual features such as grid lines, grid cells, and game states using OpenCV and other image processing techniques. After that, it runs a MobileNet Convolutional Neural Network (CNN) to output cell probabilities and feed that to a RL algorithm. Finally, the policy is trained using Proximal Policy Optimization (PPO) from stable-baseline3 on a custom built environment.

## Reinforcement Learning Architecture

### Custom Environment
- **Observation Space**: Probabilistic matrix representing game state
- **Action Space**: Discrete actions for snake movement
- **State Representation**: Robust probabilistic grid with noise injection
- **Reward System**: Distance-based rewards using Manhattan distance

### Neural Network Architecture
- **Feature Extractor**: CNN with adaptive pooling for any board size
- **Policy Head**: Fully connected layers for action selection
- **Value Head**: Separate network for state value estimation

### Training Algorithm
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Policy Type**: CNN-based policy network
- **Key Features**: Adaptive learning rates and exploration incentives

## Training Methodology

### Curriculum Learning
Progressive training approach:
- Start with small boards to learn basic mechanics
- Gradually increase board size as agent improves
- Transfer learning between board sizes
- Adaptive learning rates based on complexity

### Reward Engineering
- Sparse rewards for fruit consumption
- Dense rewards for distance improvement
- Penalties for collisions and inefficiency
- Survival bonuses for prolonged gameplay

## Performance Features
- Board-size agnostic architecture
- Real-time inference capability
- Robust to visual noise and variations
- Transfer learning across different game configurations
