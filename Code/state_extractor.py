import numpy as np

def get_state_from_board(board):
    head = board["head"]
    fruit = board["fruit"]
    body = set(board["body"])
    direction = board["direction"]

    rows, cols, _ = board["prob"].shape

    if (fruit is None): fruit = (0, 0)
    if (head is None): head = (0, 0)

    i, j = head

    # Neighbor cells
    up    = (i-1, j)
    down  = (i+1, j)
    left  = (i, j-1)
    right = (i, j+1)

    def is_danger(cell):
        ci, cj = cell
        if ci < 0 or cj < 0 or ci >= rows or cj >= cols:
            return True
        return (ci,cj) in body

    # Danger in each direction relative to heading
    if direction == "UP":
        danger_straight = is_danger(up)
        danger_right    = is_danger(right)
        danger_left     = is_danger(left)
    elif direction == "DOWN":
        danger_straight = is_danger(down)
        danger_right    = is_danger(left)
        danger_left     = is_danger(right)
    elif direction == "LEFT":
        danger_straight = is_danger(left)
        danger_right    = is_danger(up)
        danger_left     = is_danger(down)
    else:  # RIGHT
        danger_straight = is_danger(right)
        danger_right    = is_danger(down)
        danger_left     = is_danger(up)

    # Direction one-hot
    dir_l = direction == "LEFT"
    dir_r = direction == "RIGHT"
    dir_u = direction == "UP"
    dir_d = direction == "DOWN"

    # Fruit relationships
    fruit_i, fruit_j = fruit

    fruit_left  = fruit_j < j
    fruit_right = fruit_j > j
    fruit_up    = fruit_i < i
    fruit_down  = fruit_i > i

    return np.array([
        danger_straight,
        danger_right,
        danger_left,

        dir_l, dir_r, dir_u, dir_d,

        fruit_left,
        fruit_right,
        fruit_up,
        fruit_down
    ], dtype=int)
