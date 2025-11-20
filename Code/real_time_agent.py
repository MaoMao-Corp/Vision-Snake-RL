import time
import torch
import pyautogui
import numpy as np

from agent import Agent               # your Agent class
from model import Linear_QNet         # your model architecture
from board_extractor import board_stream
from state_extractor import get_state_from_board

# ---- Settings ----
MODEL_PATH = "Code/model/model_dql_final (Baseline).pth"
USE_DIST_FEATURES = False   # must match the model you trained
CONTROL_FPS = 30            # keypress frequency

pyautogui.FAILSAFE = False


# -------------------------
# Load trained model
# -------------------------
def load_pretrained_agent():
    print("[INFO] Loading trained RL model:", MODEL_PATH)

    # Must match input size of the model you trained
    input_size = 18 if USE_DIST_FEATURES else 11

    model = Linear_QNet(input_size, 256, 3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("[INFO] RL Model Loaded Successfully.")
    return model


# -------------------------
# Convert model → simple action selection
# -------------------------
def select_action(model, state):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32)
        prediction = model(state_t)
        action = torch.argmax(prediction).item()
    return action


# -------------------------
# Actual keypress execution
# -------------------------
def execute_action(action, current_direction):
    """
    Our RL outputs:
    0 = turn left
    1 = straight
    2 = turn right

    You must track direction yourself.
    """

    # convert direction enum to arrows
    DIR_ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]

    idx = DIR_ORDER.index(current_direction)

    if action == 0:   # left
        new_dir = DIR_ORDER[(idx - 1) % 4]
    elif action == 2: # right
        new_dir = DIR_ORDER[(idx + 1) % 4]
    else:
        new_dir = current_direction

    # press corresponding arrow
    if new_dir == "UP":
        pyautogui.press("up")
    elif new_dir == "DOWN":
        pyautogui.press("down")
    elif new_dir == "LEFT":
        pyautogui.press("left")
    elif new_dir == "RIGHT":
        pyautogui.press("right")


    return new_dir

# -------------------------
# Real-time main loop
# -------------------------w
def run_agent():
    model = load_pretrained_agent()
    direction = "RIGHT"     # assume initial direction

    for board in board_stream(visualize=True):

        # Fill in direction for state extractor
        board["direction"] = direction

        # Convert board to RL state vector
        state = get_state_from_board(board)

        # Model chooses action
        action = select_action(model, state)

        # Execute physical keypress
        direction = execute_action(action, direction)

        time.sleep(1.0 / CONTROL_FPS)
 
if __name__ == "__main__":
    print("\nStarting Real-Time Snake Agent...\n")
    run_agent()
