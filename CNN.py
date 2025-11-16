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
# Main Training Pipeline
###########################################

from torch.utils.data import WeightedRandomSampler

if __name__ == "__main__":
    full_dataset = datasets.ImageFolder("a_dataset")
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)

    # Get class counts from the full dataset
    class_counts = Counter([y for _, y in full_dataset.samples])
    print("Class counts:", class_counts)
    
    # Compute class weights for loss function
    total = sum(class_counts.values())
    weights = torch.tensor([
        total / (num_classes * class_counts[c]) for c in range(num_classes)
    ], dtype=torch.float32)
    print("Class weights for loss:", weights)

    # Train/validation split FIRST
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Apply transforms
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    # Compute weights for sampler based on TRAINING SET only
    train_targets = [train_ds.dataset.targets[i] for i in train_ds.indices]
    train_class_counts = Counter(train_targets)
    print("Train class counts:", train_class_counts)
    
    # Create sample weights for WeightedRandomSampler
    sample_weights = [1.0 / train_class_counts[label] ** 0.5 for label in train_targets]
    
    # Create WeightedRandomSampler for training set only
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(train_ds),  # Use actual training set size
                                    replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)  # No shuffle for validation

    # Model
    model = MobileNet(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Weighted loss - use the weights computed earlier
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    torch.save(model.state_dict(), "mobilenet_mini_snake.pth")
    print("Model saved.")

    from sklearn.metrics import confusion_matrix

    # After training, compute confusion matrix on validation set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds)

    # Print with class names
    print("\nConfusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print("Classes:", class_names)
    print(cm)

    # Optional: prettier print
    print("\nConfusion Matrix (readable):")
    header = "Pred   " + " ".join(f"{c:>6}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{x:>6}" for x in row)
        print(f"Actual {class_names[i]:<6} {row_str}")


    # After training, compute inference speed
    print("\n" + "="*50)
    print("COMPUTING INFERENCE SPEED")
    print("="*50)
    
    # Load your trained model
    model = MobileNet(num_classes=4)  # Or your enhanced version
    model.load_state_dict(torch.load("mobilenet_mini_snake.pth"))
    model.to(device)
    model.eval()
    
    # Compute speed with different batch sizes
    batch_sizes = [1, 8, 32]
    
    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")
        compute_inference_speed(
            model=model,
            device=device,
            input_size=(batch_size, 3, 64, 64),  # Adjust to your input size
            num_runs=500 if batch_size > 1 else 1000
        )