# Screen Capture RL Snake Player

This project uses computer vision techniques to capture and process the game screen of a Snake game, making it possible to apply reinforcement learning (RL) to control the snake. The screen capture part of the system extracts important visual features such as grid lines, grid cells, and game states using OpenCV and other image processing techniques. This is a crucial step for building an RL model that can play the Snake game based on visual input.

## Table of Contents

- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [Screen Capture Functionality](#screen-capture-functionality)
- [Preprocessing](#preprocessing)
- [Hough Line Detection](#hough-line-detection)
- [Grid Detection and Cell Saving](#grid-detection-and-cell-saving)
- [Run the Script](#run-the-script)

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3D24?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/rl-snake-player.git
   cd rl-snake-player
