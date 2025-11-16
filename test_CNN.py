import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from PIL import Image
import torch.nn.functional as F
import time
import torch
from thop import profile  # For FLOPs calculation

###########################################
# Speed
###########################################


def compute_inference_speed(model, device, input_size=(1, 3, 64, 64), num_runs=1000):
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup (important for accurate timing)
    print("Warming up...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    
    # Measure inference time
    print(f"Measuring inference speed over {num_runs} runs...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    end_time = time.time()
    
    # Calculate statistics
    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs * 1000  # Convert to milliseconds
    fps = num_runs / total_time  # Frames per second
    
    # Compute FLOPs and parameters
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    print("\n=== INFERENCE SPEED RESULTS ===")
    print(f"Average inference time: {avg_time_per_run:.2f} ms")
    print(f"Frames per second (FPS): {fps:.2f}")
    print(f"Model FLOPs: {flops / 1e6:.2f} MFLOPs")
    print(f"Model parameters: {params / 1e3:.2f} K")
    print(f"Total measurement time: {total_time:.2f} seconds")
    
    return avg_time_per_run, fps, flops, params


###########################################
# Pixelation Augmentation
###########################################

class RandomPixelate:
    def __init__(self, p=0.3, min_ratio=0.5, max_ratio=0.8):
        self.p = p
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img
        
        w, h = img.size
        ratio = torch.empty(1).uniform_(self.min_ratio, self.max_ratio).item()

        small_w = max(1, int(w * ratio))
        small_h = max(1, int(h * ratio))

        img_small = img.resize((small_w, small_h), Image.BILINEAR)
        img_large = img_small.resize((w, h), Image.NEAREST)
        return img_large


###########################################
# Training + Validation Transforms
###########################################

train_transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),

    # Symmetries
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    # Rotations including 45° multiples
    transforms.RandomRotation(degrees=[0, 315]),

    # Color augmentations
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1,
    ),

    RandomPixelate(),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


###########################################
# MobileNet Mini Model
###########################################

class EnhancedDepthwiseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4):
        super().__init__()
        # Expansion layer
        expanded_ch = in_ch * expansion
        self.expand = nn.Conv2d(in_ch, expanded_ch, 1) if expansion > 1 else nn.Identity()
        self.bn1 = nn.BatchNorm2d(expanded_ch)
        
        # Depthwise
        self.dw = nn.Conv2d(expanded_ch, expanded_ch, 3, stride=1, padding=1, groups=expanded_ch)
        self.bn2 = nn.BatchNorm2d(expanded_ch)
        
        # Projection
        self.pw = nn.Conv2d(expanded_ch, out_ch, 1)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.expansion = expansion

    def forward(self, x):
        if self.expansion > 1:
            x = F.relu(self.bn1(self.expand(x)))
        x = F.relu(self.bn2(self.dw(x)))
        x = self.bn3(self.pw(x))
        return F.relu(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Initial conv layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Enhanced blocks with residual connections
        self.block1 = EnhancedDepthwiseBlock(32, 32, expansion=2)
        self.block2 = EnhancedDepthwiseBlock(32, 64, expansion=4)
        self.block3 = EnhancedDepthwiseBlock(64, 128, expansion=4)
        self.block4 = EnhancedDepthwiseBlock(128, 256, expansion=2)
        
        # Additional block for more capacity
        self.block5 = EnhancedDepthwiseBlock(256, 512, expansion=4)
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)                      # 64→32
        x = F.max_pool2d(self.block1(x), 2)   # 32→16
        x = F.max_pool2d(self.block2(x), 2)   # 16→8
        x = F.max_pool2d(self.block3(x), 2)   # 8→4
        x = self.block4(x)                    # 4→4
        x = self.block5(x)                    # 4→4
        
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


###########################################
# Training + Validation Loops
###########################################

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for imgs, labels in tqdm(loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validation"):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            total += imgs.size(0)

    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / total, correct / total, f1

###########################################
# Single Image Testing Pipeline
###########################################

import os
from PIL import Image

def test_single_image(model, device, image_path, class_names, transform):
    """Test a single image and return probabilities"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    # Convert to readable format
    probs = probabilities[0].cpu().numpy()
    predicted_class_idx = probs.argmax()
    predicted_class = class_names[predicted_class_idx]
    
    return probs, predicted_class, original_image

def print_probabilities(probs, class_names):
    """Print probabilities in a nice format"""
    print("\n" + "="*50)
    print("CLASS PROBABILITIES")
    print("="*50)
    for i, (class_name, prob) in enumerate(zip(class_names, probs)):
        print(f"{class_name:>10}: {prob:.4f} ({prob*100:6.2f}%)")
    
    # Highlight prediction
    predicted_idx = probs.argmax()
    print("-" * 50)
    print(f"{'PREDICTION':>10}: {class_names[predicted_idx]} "
          f"(confidence: {probs[predicted_idx]*100:.2f}%)")

if __name__ == "__main__":
    # Define class names (must match your training)
    class_names = ['body', 'empty', 'fruit', 'head']  # Adjust if different
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    model = MobileNet(num_classes=len(class_names))
    model.load_state_dict(torch.load("mobilenet_mini_snake.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Validation transform (same as used during validation)
    val_transform = transforms.Compose([
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                           std=[0.5, 0.5, 0.5]),
    ])
    
    # Test on a specific image
    image_path = "a_dataset/fruit/cell_0_1h.png"  # Change this to your image path
    
    if not os.path.exists(image_path):
        print(f"\nError: Image file '{image_path}' not found!")
        print("Please specify a valid image path.")
    else:
        print(f"\nTesting image: {image_path}")
        
        # Get predictions
        probs, predicted_class, image = test_single_image(
            model, device, image_path, class_names, val_transform
        )
        
        # Display results
        print(f"\nImage size: {image.size}")
        print_probabilities(probs, class_names)
        
        # Optional: Show top-2 predictions
        print("\n" + "="*30)
        print("TOP PREDICTIONS")
        print("="*30)
        sorted_indices = probs.argsort()[::-1]  # Sort descending
        for i, idx in enumerate(sorted_indices[:2]):  # Top 2
            print(f"{i+1}. {class_names[idx]:>10}: {probs[idx]:.4f} ({probs[idx]*100:6.2f}%)")