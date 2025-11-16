import mss
import numpy as np
import cv2
import os

# -------------------------
# SCREENSHOT FUNCTION
# -------------------------
def screenshot(x=550, y=250):
    with mss.mss() as sct:
        monitor = sct.monitors[2]  # choose your monitor

        region = {
            "top": monitor["top"] + y,
            "left": monitor["left"] + x,
            "width": 820,
            "height": 750
        }

        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# -------------------------
# PREPROCESSING
# -------------------------
def preprocessing(img, threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    sobel_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_mag = cv2.convertScaleAbs(sobel_mag)

    sobel_binary = (sobel_mag > threshold).astype(np.uint8) * 255
    return equalized, sobel_binary

# -------------------------
# HOUGH LINE DETECTION
# -------------------------
def detect_hough_lines(sobel_binary, max_line_gap=5):
    lines = cv2.HoughLinesP(sobel_binary, rho=1, theta=np.pi/180,
                            threshold=100, minLineLength=sobel_binary.shape[0] / 4,
                            maxLineGap=max_line_gap)
    if lines is None:
        return [], []

    # Only keep horizontal / vertical lines
    h_lines, v_lines = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if abs(y1 - y2) <= 1:  # horizontal
            h_lines.append(l[0])
        elif abs(x1 - x2) <= 1:  # vertical
            v_lines.append(l[0])
    return h_lines, v_lines

# -------------------------
# MERGE NEARBY LINES
# -------------------------
def merge_lines(lines, axis=0, threshold=20):
    if not lines:
        return []

    lines = sorted(lines, key=lambda l: l[axis])
    merged = [lines[0]]
    for l in lines[1:]:
        if abs(l[axis] - merged[-1][axis]) <= threshold:
            merged[-1] = [(a + b) / 2 for a, b in zip(merged[-1], l)]
        else:
            merged.append(l)
    return merged

# -------------------------
# DRAW GRID POINTS
# -------------------------
def draw_grid(img, grid):
    img = img.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            y, x = grid[i, j]
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, f"({i},{j})", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
    return img


import numpy as np

# -------------------------
# TRIM EDGE LINES
# -------------------------
def trimLines(h_lines, v_lines, distance_thresh=0):
    # Horizontal lines → remove lines whose x2 is within distance_thresh of the rightmost x2
    if h_lines:
        max_x2 = max(l[3] for l in h_lines)
        print(max_x2)
        trimmed_h_lines = [l for l in h_lines if (max_x2 - l[3]) > distance_thresh]
    else:
        trimmed_h_lines = []

    # Vertical lines → remove lines whose y2 is within distance_thresh of the bottommost y2
    if v_lines:
        max_y2 = max(l[2] for l in v_lines)
        trimmed_v_lines = [l for l in v_lines if (max_y2 - l[2]) > distance_thresh]
    else:
        trimmed_v_lines = []

    return trimmed_h_lines, trimmed_v_lines

# -------------------------
# SAVE CELLS
# -------------------------
import random

def save_cells(img, grid, cell_size, output_folder="cells", extra=0.1, particular = None):
    os.makedirs(output_folder, exist_ok=True)
    cell_size = int(cell_size)
    rows, cols, _ = grid.shape

    extra_pixl = int(cell_size * extra)

    if (particular is not None):
        for i, j in particular:
            y, x = grid[i, j]
            # center crop around the grid point
            y1 = max(0, y - extra_pixl)
            x1 = max(0, x - extra_pixl)
            y2 = min(y1 + cell_size + 2*extra_pixl, img.shape[0])
            x2 = min(x1 + cell_size + 2*extra_pixl, img.shape[1])

            cell_img = img[y1:y2, x1:x2]

            filename = os.path.join(output_folder, f"cell_{i}_{j}_{random.randint(0,1500)}.png")
            cv2.imwrite(filename, cell_img)
    else:
        for i in range(rows):
            for j in range(cols):
                y, x = grid[i, j]

                # center crop around the grid point
                y1 = max(0, y - extra_pixl)
                x1 = max(0, x - extra_pixl)
                y2 = min(y1 + cell_size + 2*extra_pixl, img.shape[0])
                x2 = min(x1 + cell_size + 2*extra_pixl, img.shape[1])

                cell_img = img[y1:y2, x1:x2]

                filename = os.path.join(output_folder, f"cell_{i}_{j}.png")
                cv2.imwrite(filename, cell_img)

    print(f"Saved {rows*cols} cells in '{output_folder}'")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    img = screenshot()
    equalized, sobel_binary = preprocessing(img)

    h_lines, v_lines = detect_hough_lines(sobel_binary)
    h_lines = merge_lines(h_lines, axis=1)
    v_lines = merge_lines(v_lines, axis=0)

    h_lines, v_lines = trimLines(h_lines, v_lines)

    h_lines_sorted = sorted(h_lines, key=lambda l: l[1])  # sort by y
    v_lines_sorted = sorted(v_lines, key=lambda l: l[0])  # sort by x

    grid = np.zeros((len(h_lines_sorted), len(v_lines_sorted), 2), dtype=int)
    
    # Rough intersections from lines

    for i, h in enumerate(h_lines_sorted):
        y = int(h[1])
        for j, v in enumerate(v_lines_sorted):
            x = int(v[0])
            grid[i, j] = (y, x)

    # Estimate cell size from horizontal avg
    cell_size = np.mean(np.diff(grid[0, :, 1]))
    print("Cell size:", cell_size)

    # Draw the grid
    img_grid = draw_grid(img, grid)

    cellList = [(4,2)]

    # Display
    cv2.imshow("Aligned Regular Grid", img_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_cells(img, grid, int(cell_size), output_folder="a_dataset/fruit2", particular=cellList)
