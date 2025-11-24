import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from vision import screenshot, preprocessing, detect_hough_lines
from vision import merge_lines, trimLines
from test_CNN import MobileNet

# -------------------------
# SETTINGS
# -------------------------
CELL_EXTRA = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['body', 'empty', 'fruit', 'head']
MODEL_PATH = "Code/mobilenet_mini_snake.pth"
CELL_SIZE_ESTIMATE = None

# BGR colors for visualization (optional)
CLASS_COLORS = {
    'body':  (0, 255, 0),
    'empty': (50, 50, 50),
    'fruit': (0, 0, 255),
    'head':  (255, 0, 0),
}

# -------------------------
# UTILS
# -------------------------
def image_changed(img1, img2, threshold=10):
    if img1 is None or img2 is None:
        return True
    if img1.shape != img2.shape:
        return True
    diff = cv2.absdiff(img1, img2)
    return np.mean(diff) > threshold

def crop_cell(img, y, x, cell_size, extra=CELL_EXTRA):
    cell_size = int(cell_size)
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

print("[INFO] Model loaded on", DEVICE)

# -------------------------
# TRANSFORM
# -------------------------
val_transform = transforms.Compose([
    transforms.Resize((64, 64), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


# ============================================================
#   MAIN BOARD INITIALIZATION (GRID DETECTION)
# ============================================================
def initialize_grid():
    global CELL_SIZE_ESTIMATE

    img = screenshot(x=1030, y=250)
    _, sobel = preprocessing(img)

    h_lines, v_lines = detect_hough_lines(sobel)
    h_lines = merge_lines(h_lines, axis=1)
    v_lines = merge_lines(v_lines, axis=0)

    h_lines, v_lines = trimLines(h_lines, v_lines)

    h_sort = sorted(h_lines, key=lambda h: h[1])
    v_sort = sorted(v_lines, key=lambda v: v[0])

    grid = np.zeros((len(h_sort), len(v_sort), 2), dtype=int)

    for i, h in enumerate(h_sort):
        y = int(h[1])
        for j, v in enumerate(v_sort):
            x = int(v[0])
            grid[i, j] = (y, x)

    # estimate pixel size horizontally
    CELL_SIZE_ESTIMATE = np.mean(np.diff(grid[0, :, 1]))
    print("[INFO] Cell size =", CELL_SIZE_ESTIMATE)

    return grid


def head_candidate_score(i, j, prob_matrix):
    """
    Higher score -> more likely this is the head.
    Combines:
      - prob_matrix[i,j,head_idx] (CNN confidence)
      - favors cells with only 1 neighbor with high body/head probability (extreme)
    """
    head_idx = CLASS_NAMES.index("head")
    body_idx = CLASS_NAMES.index("body")
    
    score = prob_matrix[i, j, head_idx]
    
    neighbors = 0
    rows, cols, _ = prob_matrix.shape
    for di,dj in [(0,1),(0,-1),(1,0),(-1,0)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < rows and 0 <= nj < cols:
            neighbors += prob_matrix[ni, nj, head_idx] + prob_matrix[ni, nj, body_idx]
    
    # endpoints have fewer neighbors → divide by (neighbors+1)
    score /= (neighbors + 1e-6)
    return score

from state_corrector import SnakeStateCorrector

# ============================================================
#   FRAME → BOARD DICT GENERATOR
# ============================================================
def board_stream(visualize=True):
    """
    Yields a dict each frame:

    {
        "prob": prob_matrix,   # (rows, cols, 4)
        "head": (i,j),
        "fruit": (i,j),
        "body": [(i,j), ...],
        "empty": [(i,j), ...],
        "direction": None
    }
    """

    grid = initialize_grid()
    rows, cols, _ = grid.shape

    snake_corrector = SnakeStateCorrector(rows, cols)

    # probability matrix
    prob_matrix = np.zeros((rows, cols, len(CLASS_NAMES)))
    last_cell_images = [[None for _ in range(cols)] for _ in range(rows)]

    while True:

        frame = screenshot(x=1030, y=250)

        # PROCESS CELLS
        for i in range(rows):
            for j in range(cols):
                y, x = grid[i, j]
                cell_img = crop_cell(frame, y, x, CELL_SIZE_ESTIMATE)

                # do inference only if changed
                if image_changed(cell_img, last_cell_images[i][j]):
                    last_cell_images[i][j] = cell_img

                    pil = Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
                    tensor = val_transform(pil).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        out = model(tensor)
                        probs = F.softmax(out, dim=1).cpu().numpy()[0]

                    prob_matrix[i, j] = probs

        # ENFORCE ONE HEAD / ONE FRUIT
        # Step 0: force fruit
        fruit_idx = CLASS_NAMES.index("fruit")
        flat = prob_matrix[:, :, fruit_idx]
        fruit_pos = np.unravel_index(np.argmax(flat), (rows, cols))
        prob_matrix[:, :, fruit_idx] = 0  # zero out all fruit scores
        prob_matrix[fruit_pos[0], fruit_pos[1], fruit_idx] = 1.0
        
        # Step 1: pick head
        head_idx = CLASS_NAMES.index("head")
        best_score = -1
        max_idx = None

        # Use a temporary copy for scoring so forced cells don't influence neighbors
        temp_matrix = prob_matrix.copy()

        for i in range(rows):
            for j in range(cols):
                if (i,j) == fruit_pos:
                    continue
                s = head_candidate_score(i, j, temp_matrix)
                if s > best_score:
                    best_score = s
                    max_idx = (i, j)

        # Step 2: force head
        prob_matrix[:, :, head_idx] = 0  # zero out all head probabilities
        prob_matrix[max_idx[0], max_idx[1], head_idx] = 1.0

        # BUILD CELL LISTS
        head_pos = None
        fruit_pos = None
        body_cells = []
        empty_cells = []

        for i in range(rows):
            for j in range(cols):
                c_idx = prob_matrix[i, j].argmax()
                cname = CLASS_NAMES[c_idx]

                if cname == "head":
                    head_pos = (i, j)
                elif cname == "fruit":
                    fruit_pos = (i, j)
                elif cname == "body":
                    body_cells.append((i, j))
                else:
                    empty_cells.append((i, j))

        # PACKAGE BOARD DICTIONARY
        board_dict = {
            "prob": prob_matrix.copy(),
            "head": head_pos,
            "fruit": fruit_pos,
            "body": body_cells,
            "empty": empty_cells,
            "direction": None,  # fill externally
        }

        corrected_board = snake_corrector.correct_board_state(board_dict)

        # VISUALIZATION
        if visualize:
            disp = frame.copy()
            cs = int(CELL_SIZE_ESTIMATE)
            ex = int(cs * CELL_EXTRA)

            for i in range(rows):
                for j in range(cols):
                    y, x = grid[i, j]
                    y1 = max(0, y - ex)
                    x1 = max(0, x - ex)
                    y2 = min(y1 + cs + 2*ex, disp.shape[0])
                    x2 = min(x1 + cs + 2*ex, disp.shape[1])

                    cname = CLASS_NAMES[prob_matrix[i, j].argmax()]
                    color = CLASS_COLORS[cname]
                    cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

            direction = None

            # Draw direction arrow on snake head
            if board_dict["head"] is not None and direction is not None:
                head_i, head_j = board_dict["head"]
                y, x = grid[head_i, head_j]
                
                # Center of the head cell
                center_x = x
                center_y = y
                
                # Arrow parameters
                arrow_length = cs // 2
                
                # Calculate arrow end point based on direction
                if direction == "UP":
                    end_x = center_x
                    end_y = center_y - arrow_length
                elif direction == "DOWN":
                    end_x = center_x
                    end_y = center_y + arrow_length
                elif direction == "LEFT":
                    end_x = center_x - arrow_length
                    end_y = center_y
                elif direction == "RIGHT":
                    end_x = center_x + arrow_length
                    end_y = center_y
                else:
                    end_x = center_x
                    end_y = center_y
                
                # Draw arrow (thicker and more visible)
                cv2.arrowedLine(disp, 
                                (center_x, center_y), 
                                (end_x, end_y), 
                                color=(255, 255, 0),  # Bright yellow
                                thickness=3,
                                tipLength=0.3)
                
                # Optional: Add a small circle at head center for better visibility
                cv2.circle(disp, (center_x, center_y), 4, (255, 255, 255), -1)  # White dot

            cv2.imshow("Board Extractor", disp)
            if cv2.waitKey(1) == 27:
                break

        yield corrected_board

    cv2.destroyAllWindows()
