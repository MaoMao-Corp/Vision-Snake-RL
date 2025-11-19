import numpy as np
from collections import deque
import random
import gymnasium as gym
from gymnasium import spaces

class SnakeGame:
    """
    Deterministic Snake game logic:
    - Board
    - Snake's body
    - Fruit
    - Directions
    """

    UP = np.array([-1, 0])
    DOWN = np.array([1, 0])
    LEFT = np.array([0, -1])
    RIGHT = np.array([0, 1])

    DIRECTIONS = [UP, RIGHT, DOWN, LEFT]  # clockwise order

    def __init__(self, size=15):
        self.size = size
        self.reset()

    def reset(self):
        self.snake = deque()

        # center the snake
        m = self.size // 2
        self.snake.appendleft((m, m))    # head

        # start moving right
        self.dir_idx = 1  # pointing RIGHT

        # place fruit
        self._place_fruit()

        return self._get_state()

    def _place_fruit(self):
        while True:
            pos = (random.randint(0, self.size - 1),
                   random.randint(0, self.size - 1))
            if pos not in self.snake:
                self.fruit = pos
                return

    def turn_left(self):
        self.dir_idx = (self.dir_idx - 1) % 4

    def turn_right(self):
        self.dir_idx = (self.dir_idx + 1) % 4

    def step(self, action):
        """
        action: 0 = turn left, 1 = straight, 2 = turn right
        """

        if action == 0:
            self.turn_left()
        elif action == 2:
            self.turn_right()

        direction = SnakeGame.DIRECTIONS[self.dir_idx]
        head = np.array(self.snake[0]) + direction
        head = tuple(head)

        # check wall collision
        if not (0 <= head[0] < self.size and 0 <= head[1] < self.size):
            return self._get_state(), -1.0, True

        # check self collision
        if head in self.snake:
            return self._get_state(), -1.0, True

        # move snake
        self.snake.appendleft(head)

        # check fruit
        if head == self.fruit:
            reward = 1.0
            self._place_fruit()
        else:
            reward = -0.01   # time penalty
            self.snake.pop()

        return self._get_state(), reward, False

    def _get_state(self):
        return {
            "snake": list(self.snake),
            "fruit": self.fruit,
            "dir_idx": self.dir_idx
        }

class NoiseModel:
    def __init__(self, num_classes=3, gaussian_std=0.05, flip_prob=0.01):
        self.C = num_classes
        self.std = gaussian_std
        self.flip_prob = flip_prob

    def noisy_onehot(self, true_class):
        vec = np.zeros(self.C)

        # Base probability: strong confidence in true class
        vec[true_class] = 0.9 + np.random.random() * 0.1  # 0.9–1.0

        # Random flip: increase a wrong class
        if np.random.rand() < self.flip_prob:
            wrong = np.random.randint(0, self.C)
            if wrong == true_class:
                # pick a truly different class
                wrong = (wrong + np.random.randint(1, self.C)) % self.C
            vec[wrong] += 0.6

        # Add Gaussian noise
        vec += np.random.normal(0, self.std, self.C)

        # Avoid numerical explosion before softmax
        vec = np.clip(vec, -10, 10)

        # Normalize to probability distribution with softmax
        e = np.exp(vec - np.max(vec))
        vec = e / e.sum()

        return vec

class SnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    SNAKE = 0    # was 1
    FRUIT = 1    # was 2  
    HEAD = 2     # was 3
    # Removed EMPTY channel

    def __init__(self, size=15, noise=NoiseModel()):
        super().__init__()
        self.game = SnakeGame(size)
        self.size = size
        self.noise = noise

        # 3 actions: left, straight, right
        self.action_space = spaces.Discrete(3)

        # observation: now 3 channels instead of 4
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3, size, size),  # CHW format - only 3 channels now!
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state = self.game.reset()
        obs = self._make_observation(state)
        return obs, {}

    def step(self, action):
        state, reward, done = self.game.step(action)
        obs = self._make_observation(state)
        return obs, reward, done, False, {}

    def _make_observation(self, state):
        # Now only 3 channels: [SNAKE, FRUIT, HEAD]
        mat = np.zeros((self.size, self.size, 3), dtype=np.float32)

        # Fill snake body (excluding head)
        for i, (r, c) in enumerate(state["snake"]):
            if i == 0:  # Head - we'll handle separately
                continue
            mat[r, c, SnakeEnv.SNAKE] = 1.0

        # Head
        hr, hc = state["snake"][0]
        mat[hr, hc, SnakeEnv.HEAD] = 1.0

        # Fruit
        fr, fc = state["fruit"]
        mat[fr, fc, SnakeEnv.FRUIT] = 1.0

        # Apply noise tile-by-tile
        noisy = np.zeros_like(mat)
        for i in range(self.size):
            for j in range(self.size):
                true_class = np.argmax(mat[i, j])
                noisy[i, j] = self.noise.noisy_onehot(true_class)

        return noisy.transpose(2, 0, 1)  # Shape: (3, size, size)