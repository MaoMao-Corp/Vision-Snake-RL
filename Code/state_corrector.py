# state_corrector.py
import numpy as np
from collections import deque

CLASS_NAMES = ["empty", "body", "head", "fruit"]

class SnakeStateCorrector:
    def __init__(self, grid_rows, grid_cols):
        self.rows = grid_rows
        self.cols = grid_cols
        self.previous_states = deque(maxlen=5)
        self.last_fruit_pos = None
        self.initialized = False
        
    def correct_board_state(self, current_board):
        """
        Apply snake game rules with simple temporal reasoning
        """
        if not self.initialized:
            self.previous_states.append(current_board.copy())
            self.initialized = True
            return current_board
            
        # Get previous state for temporal reasoning
        prev_board = self.previous_states[-1] if self.previous_states else None
        
        # Apply correction rules
        corrected_board = self._apply_simple_rules(current_board, prev_board)
        self.previous_states.append(corrected_board.copy())
        
        return corrected_board
    
    def _apply_simple_rules(self, current_board, prev_board):
        """
        Apply simple snake movement and game rules
        """
        prob_matrix = current_board["prob"].copy()
        head_pos = current_board["head"]
        fruit_pos = current_board["fruit"]
        body_cells = current_board["body"].copy()
        direction = current_board["direction"]
        
        # Apply rules in order
        prob_matrix = self._enforce_weighted_fruit(prob_matrix, prev_board)
        prob_matrix = self._enforce_single_head(prob_matrix)
        prob_matrix, body_cells = self._enforce_body_connectivity(prob_matrix, head_pos, body_cells)
        prob_matrix = self._enforce_movement_rules(prob_matrix, head_pos, direction, prev_board)
        
        # Update the board with corrected values
        corrected_board = current_board.copy()
        corrected_board["prob"] = prob_matrix
        corrected_board["body"] = body_cells
        
        # Re-extract positions from corrected probabilities
        corrected_board = self._update_positions_from_prob(corrected_board)
        
        return corrected_board
    
    def _enforce_weighted_fruit(self, prob_matrix, prev_board):
        """Fruit takes MAX of weighted matrix prioritizing last fruit position"""
        fruit_idx = CLASS_NAMES.index("fruit")
        
        # Get current fruit probabilities
        current_fruit_probs = prob_matrix[:, :, fruit_idx].copy()
        
        # Create weighted matrix
        weighted_probs = current_fruit_probs.copy()
        
        # If we have a previous fruit position, boost its probability
        if prev_board and prev_board["fruit"]:
            last_i, last_j = prev_board["fruit"]
            
            # Check if fruit was eaten (head moved to last fruit position)
            current_head_pos = tuple(np.unravel_index(np.argmax(prob_matrix[:, :, CLASS_NAMES.index("head")]), prob_matrix.shape[:2]))
            fruit_was_eaten = (current_head_pos == (last_i, last_j))
            fruit_was_eaten = False  # Disable fruit eaten logic for simplicity

            if not fruit_was_eaten:
                # Boost probability at last fruit position
                weighted_probs[last_i, last_j] *= 3  # Strong boost
            else:
                # Fruit was eaten - don't boost last position
                # Let current probabilities decide new fruit location
                pass
        
        # Find the max in weighted matrix
        fruit_max_i, fruit_max_j = np.unravel_index(np.argmax(weighted_probs), weighted_probs.shape)
        
        # Force single fruit at max weighted position
        prob_matrix[:, :, fruit_idx] = 0
        prob_matrix[fruit_max_i, fruit_max_j, fruit_idx] = 1.0
        
        # Update last fruit position
        self.last_fruit_pos = (fruit_max_i, fruit_max_j)
        
        return prob_matrix
    
    def _enforce_single_head(self, prob_matrix):
        """Ensure exactly one head exists"""
        head_idx = CLASS_NAMES.index("head")
        
        # Find current max probabilities
        head_probs = prob_matrix[:, :, head_idx]
        
        # Force single head - keep only the highest probability head
        head_max_i, head_max_j = np.unravel_index(np.argmax(head_probs), head_probs.shape)
        prob_matrix[:, :, head_idx] = 0
        prob_matrix[head_max_i, head_max_j, head_idx] = 1.0
        
        return prob_matrix
    
    def _enforce_body_connectivity(self, prob_matrix, head_pos, body_cells):
        """Ensure snake body forms a connected path"""
        if not head_pos or len(body_cells) == 0:
            return prob_matrix, body_cells
            
        head_idx = CLASS_NAMES.index("head")
        body_idx = CLASS_NAMES.index("body")
        empty_idx = CLASS_NAMES.index("empty")
        
        # Create visited set starting from head
        visited = set()
        queue = [head_pos]
        connected_body = set()
        
        while queue:
            i, j = queue.pop(0)
            if (i, j) in visited:
                continue
                
            visited.add((i, j))
            
            # Check neighbors for body/head cells
            for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    cell_type = CLASS_NAMES[prob_matrix[ni, nj].argmax()]
                    if cell_type in ["head", "body"] and (ni, nj) not in visited:
                        queue.append((ni, nj))
                        if cell_type == "body":
                            connected_body.add((ni, nj))
        
        # Remove disconnected body cells (convert to empty)
        disconnected_bodies = set(body_cells) - connected_body
        for i, j in disconnected_bodies:
            prob_matrix[i, j, body_idx] = 0
            prob_matrix[i, j, empty_idx] = 1.0
        
        return prob_matrix, list(connected_body)
    
    def _enforce_movement_rules(self, prob_matrix, head_pos, direction, prev_board):
        """Basic movement rules"""
        if not head_pos or not direction:
            return prob_matrix
            
        head_idx = CLASS_NAMES.index("head")
        body_idx = CLASS_NAMES.index("body")
        
        i, j = head_pos
        
        # Simple rule: cell behind head should be body
        if prev_board and prev_board["head"]:
            prev_head = prev_board["head"]
            
            if prev_head != head_pos:
                # Determine the cell that should be body (where head came from)
                if direction == "RIGHT" and j > 0:
                    expected_body = (i, j-1)
                elif direction == "LEFT" and j < self.cols-1:
                    expected_body = (i, j+1)
                elif direction == "DOWN" and i > 0:
                    expected_body = (i-1, j)
                elif direction == "UP" and i < self.rows-1:
                    expected_body = (i+1, j)
                else:
                    expected_body = None
                
                # Force the expected body cell
                if expected_body:
                    ei, ej = expected_body
                    current_type = CLASS_NAMES[prob_matrix[ei, ej].argmax()]
                    if current_type != "head":  # Don't overwrite head
                        prob_matrix[ei, ej, :] = 0
                        prob_matrix[ei, ej, body_idx] = 1.0
        
        return prob_matrix
    
    def _update_positions_from_prob(self, board):
        """Update positions based on corrected probabilities"""
        prob_matrix = board["prob"]
        head_idx = CLASS_NAMES.index("head")
        fruit_idx = CLASS_NAMES.index("fruit")
        body_idx = CLASS_NAMES.index("body")
        empty_idx = CLASS_NAMES.index("empty")
        
        # Find head position
        head_probs = prob_matrix[:, :, head_idx]
        head_pos = np.unravel_index(np.argmax(head_probs), head_probs.shape)
        
        # Find fruit position  
        fruit_probs = prob_matrix[:, :, fruit_idx]
        fruit_pos = np.unravel_index(np.argmax(fruit_probs), fruit_probs.shape)
        
        # Find body cells
        body_cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                if prob_matrix[i, j].argmax() == body_idx:
                    body_cells.append((i, j))
        
        # Find empty cells
        empty_cells = []
        for i in range(self.rows):
            for j in range(self.cols):
                if prob_matrix[i, j].argmax() == empty_idx:
                    empty_cells.append((i, j))
        
        board["head"] = head_pos
        board["fruit"] = fruit_pos
        board["body"] = body_cells
        board["empty"] = empty_cells
        
        return board