import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import time
import os
from vision import screenshot, preprocessing, detect_hough_lines, merge_lines, trimLines, draw_grid
from test_CNN import MobileNet

# -------------------------
# SETTINGS
# -------------------------
CELL_EXTRA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['body', 'empty', 'fruit', 'head']  # adjust to your model
MODEL_PATH = "Code/mobilenet_mini_snake.pth"
CELL_SIZE_ESTIMATE = None  # will be computed from grid

# Define colors for each class (BGR)
CLASS_COLORS = {
    'body': (0, 255, 0),   # green
    'empty': (50, 50, 50), # gray
    'fruit': (0, 0, 255),  # red
    'head': (255, 0, 0),   # blue
}


# -------------------------
# UTILS
# -------------------------
def image_changed(img1, img2, threshold=10):
    """Return True if images differ significantly"""
    if img1.shape != img2.shape:
        return True
    diff = cv2.absdiff(img1, img2)
    return np.mean(diff) > threshold

def crop_cell(img, y, x, cell_size, extra=CELL_EXTRA):
    cell_size = int(cell_size)  # <--- ensure it's an integer
    extra_pix = int(cell_size * extra)
    y1 = max(0, y - extra_pix)
    x1 = max(0, x - extra_pix)
    y2 = min(y1 + cell_size + 2*extra_pix, img.shape[0])
    x2 = min(x1 + cell_size + 2*extra_pix, img.shape[1])
    return img[y1:y2, x1:x2]


# -------------------------
# LOAD MODEL
# -------------------------
model = MobileNet(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model loaded on", DEVICE)

# -------------------------
# TRANSFORM
# -------------------------
val_transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

# -------------------------
# INITIAL GRID DETECTION
# -------------------------
img = screenshot()
equalized, sobel_binary = preprocessing(img)
h_lines, v_lines = detect_hough_lines(sobel_binary)
h_lines = merge_lines(h_lines, axis=1)
v_lines = merge_lines(v_lines, axis=0)
h_lines, v_lines = trimLines(h_lines, v_lines)
h_lines_sorted = sorted(h_lines, key=lambda l: l[1])
v_lines_sorted = sorted(v_lines, key=lambda l: l[0])

grid = np.zeros((len(h_lines_sorted), len(v_lines_sorted), 2), dtype=int)
for i, h in enumerate(h_lines_sorted):
    y = int(h[1])
    for j, v in enumerate(v_lines_sorted):
        x = int(v[0])
        grid[i, j] = (y, x)

CELL_SIZE_ESTIMATE = np.mean(np.diff(grid[0, :, 1]))
print("Cell size estimate:", CELL_SIZE_ESTIMATE)

# -------------------------
# INITIALIZE PROBABILITY MATRIX
# -------------------------
rows, cols, _ = grid.shape
prob_matrix = np.zeros((rows, cols, len(CLASS_NAMES)))
last_cell_images = [[None for _ in range(cols)] for _ in range(rows)]

# -------------------------
# REAL-TIME LOOP
# -------------------------
try:
    while True:
        frame = screenshot()
        updated_cells = 0

        for i in range(rows):
            for j in range(cols):
                y, x = grid[i, j]
                cell_img = crop_cell(frame, y, x, CELL_SIZE_ESTIMATE)

                # Check if cell changed
                if last_cell_images[i][j] is None or image_changed(cell_img, last_cell_images[i][j]):
                    last_cell_images[i][j] = cell_img

                    # Convert to PIL and transform
                    pil_img = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
                    input_tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = F.softmax(output, dim=1).cpu().numpy()[0]

                    prob_matrix[i, j] = probs
                    updated_cells += 1

        # Ensure at least one head and one fruit
        for special_class in ['head', 'fruit']:
            class_idx = CLASS_NAMES.index(special_class)
            # Find the cell with the maximum probability for this class
            max_prob = -1
            max_cell = None
            for i in range(rows):
                for j in range(cols):
                    if prob_matrix[i, j, class_idx] > max_prob:
                        max_prob = prob_matrix[i, j, class_idx]
                        max_cell = (i, j)
            if max_cell:
                i, j = max_cell
                # Force the predicted class to be this special class
                prob_matrix[i, j] = np.zeros(len(CLASS_NAMES))
                prob_matrix[i, j, class_idx] = 1.0

        # Optional: Display grid with class color borders
        display_img = frame.copy()
        for i in range(rows):
            for j in range(cols):
                y, x = grid[i, j]
                class_idx = prob_matrix[i, j].argmax()
                class_name = CLASS_NAMES[class_idx]
                color = CLASS_COLORS[class_name]

                # Draw rectangle around cell
                cell_size_int = int(CELL_SIZE_ESTIMATE)
                extra_pix = int(cell_size_int * CELL_EXTRA)
                y1 = max(0, y - extra_pix)
                x1 = max(0, x - extra_pix)
                y2 = min(y1 + cell_size_int + 2*extra_pix, display_img.shape[0])
                x2 = min(x1 + cell_size_int + 2*extra_pix, display_img.shape[1])

                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)

        cv2.imshow("Snake Grid Probabilities", display_img)


        key = cv2.waitKey(50)  # ~20 FPS
        if key == 27:  # ESC to exit
            break

finally:
    cv2.destroyAllWindows()
