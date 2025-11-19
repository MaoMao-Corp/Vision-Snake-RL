import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SnakePPONet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
        )
        
        # Global average pooling - efficient and works great!
        self.policy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(), 
            nn.Linear(128, 4)  # Actions
        )
        
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), 
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Value
        )
    
    def forward(self, x):
        features = self.conv_layers(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value