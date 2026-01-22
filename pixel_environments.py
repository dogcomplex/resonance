"""
PIXEL ENVIRONMENTS

Simulated Atari-like environments for testing pixel-based rule learning.
These capture the key challenges without needing the actual Atari ROMs:

1. PONG: Ball bounces, paddle moves, score changes
2. BREAKOUT: Ball breaks bricks, paddle catches
3. SPACE_INVADERS: Enemies move, player shoots, enemies shoot back

Key challenge: Raw pixels → meaningful tokens → rules

Different tokenization strategies to test:
1. RAW: Every pixel is a token (huge state space)
2. GRID: Divide into NxN regions, summarize each
3. DIFF: Only track what changed between frames
4. OBJECTS: Try to detect connected components
"""

import numpy as np
import random
from typing import Set, Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class GameObject:
    """A simple game object with position and size."""
    x: float
    y: float
    w: int
    h: int
    color: int
    vx: float = 0
    vy: float = 0


class PixelPong:
    """
    Simple Pong with pixel observations.
    
    Screen: 84x84 (Atari-like)
    Objects: Ball (2x2), Paddle (2x10), Walls
    Actions: 0=stay, 1=up, 2=down
    """
    
    WIDTH = 84
    HEIGHT = 84
    PADDLE_H = 10
    PADDLE_W = 2
    BALL_SIZE = 2
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)
        
        # Paddle on right side
        self.paddle_y = self.HEIGHT // 2 - self.PADDLE_H // 2
        
        # Ball in center
        self.ball_x = self.WIDTH // 2
        self.ball_y = self.HEIGHT // 2
        self.ball_vx = -2 if self.rng.random() < 0.5 else 2
        self.ball_vy = self.rng.uniform(-1.5, 1.5)
        
        self.score = 0
        self.done = False
        
        return self._render()
    
    def _render(self) -> np.ndarray:
        """Render to 84x84 grayscale image."""
        screen = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        
        # Draw paddle (white = 255)
        px, py = self.WIDTH - 5, int(self.paddle_y)
        screen[max(0,py):min(self.HEIGHT,py+self.PADDLE_H), 
               max(0,px):min(self.WIDTH,px+self.PADDLE_W)] = 255
        
        # Draw ball (white = 255)
        bx, by = int(self.ball_x), int(self.ball_y)
        screen[max(0,by):min(self.HEIGHT,by+self.BALL_SIZE),
               max(0,bx):min(self.WIDTH,bx+self.BALL_SIZE)] = 255
        
        # Draw walls (gray = 128)
        screen[0:2, :] = 128  # Top
        screen[-2:, :] = 128  # Bottom
        screen[:, 0:2] = 128  # Left
        
        return screen
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: 0=stay, 1=up, 2=down
        """
        # Move paddle
        if action == 1:
            self.paddle_y = max(0, self.paddle_y - 3)
        elif action == 2:
            self.paddle_y = min(self.HEIGHT - self.PADDLE_H, self.paddle_y + 3)
        
        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        reward = 0
        
        # Ball collision with top/bottom
        if self.ball_y <= 2 or self.ball_y >= self.HEIGHT - 2 - self.BALL_SIZE:
            self.ball_vy = -self.ball_vy
            self.ball_y = max(2, min(self.HEIGHT - 2 - self.BALL_SIZE, self.ball_y))
        
        # Ball collision with left wall (opponent side - ball returns)
        if self.ball_x <= 2:
            self.ball_vx = abs(self.ball_vx)
            self.ball_x = 3
        
        # Ball collision with paddle
        paddle_x = self.WIDTH - 5
        if (self.ball_x + self.BALL_SIZE >= paddle_x and 
            self.ball_x <= paddle_x + self.PADDLE_W and
            self.ball_y + self.BALL_SIZE >= self.paddle_y and 
            self.ball_y <= self.paddle_y + self.PADDLE_H):
            self.ball_vx = -abs(self.ball_vx)
            self.ball_x = paddle_x - self.BALL_SIZE
            # Add spin based on where ball hit paddle
            hit_pos = (self.ball_y - self.paddle_y) / self.PADDLE_H
            self.ball_vy = (hit_pos - 0.5) * 3
            reward = 1
            self.score += 1
        
        # Ball past paddle (miss)
        if self.ball_x >= self.WIDTH - 2:
            reward = -1
            self.done = True
        
        return self._render(), reward, self.done, {'score': self.score}
    
    def get_valid_actions(self):
        return [0, 1, 2]


class PixelBreakout:
    """
    Simple Breakout with pixel observations.
    
    Screen: 84x84
    Objects: Ball, Paddle, Bricks
    Actions: 0=stay, 1=left, 2=right
    """
    
    WIDTH = 84
    HEIGHT = 84
    PADDLE_W = 12
    PADDLE_H = 2
    BALL_SIZE = 2
    BRICK_W = 8
    BRICK_H = 4
    N_BRICK_COLS = 10
    N_BRICK_ROWS = 5
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        
        # Paddle at bottom
        self.paddle_x = self.WIDTH // 2 - self.PADDLE_W // 2
        
        # Ball above paddle
        self.ball_x = self.WIDTH // 2
        self.ball_y = self.HEIGHT - 15
        self.ball_vx = self.rng.uniform(-1.5, 1.5)
        self.ball_vy = -2
        
        # Bricks
        self.bricks = []
        start_x = (self.WIDTH - self.N_BRICK_COLS * self.BRICK_W) // 2
        for row in range(self.N_BRICK_ROWS):
            for col in range(self.N_BRICK_COLS):
                self.bricks.append({
                    'x': start_x + col * self.BRICK_W,
                    'y': 10 + row * self.BRICK_H,
                    'alive': True
                })
        
        self.score = 0
        self.done = False
        
        return self._render()
    
    def _render(self) -> np.ndarray:
        screen = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        
        # Draw bricks
        for brick in self.bricks:
            if brick['alive']:
                x, y = int(brick['x']), int(brick['y'])
                screen[y:y+self.BRICK_H-1, x:x+self.BRICK_W-1] = 200
        
        # Draw paddle
        px = int(self.paddle_x)
        screen[self.HEIGHT-5:self.HEIGHT-5+self.PADDLE_H, 
               max(0,px):min(self.WIDTH,px+self.PADDLE_W)] = 255
        
        # Draw ball
        bx, by = int(self.ball_x), int(self.ball_y)
        screen[max(0,by):min(self.HEIGHT,by+self.BALL_SIZE),
               max(0,bx):min(self.WIDTH,bx+self.BALL_SIZE)] = 255
        
        return screen
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Move paddle
        if action == 1:
            self.paddle_x = max(0, self.paddle_x - 4)
        elif action == 2:
            self.paddle_x = min(self.WIDTH - self.PADDLE_W, self.paddle_x + 4)
        
        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        
        reward = 0
        
        # Wall collisions
        if self.ball_x <= 0 or self.ball_x >= self.WIDTH - self.BALL_SIZE:
            self.ball_vx = -self.ball_vx
        if self.ball_y <= 0:
            self.ball_vy = -self.ball_vy
        
        # Paddle collision
        if (self.ball_y + self.BALL_SIZE >= self.HEIGHT - 5 and
            self.ball_x + self.BALL_SIZE >= self.paddle_x and
            self.ball_x <= self.paddle_x + self.PADDLE_W):
            self.ball_vy = -abs(self.ball_vy)
            hit_pos = (self.ball_x - self.paddle_x) / self.PADDLE_W
            self.ball_vx = (hit_pos - 0.5) * 4
        
        # Brick collisions
        for brick in self.bricks:
            if not brick['alive']:
                continue
            if (self.ball_x + self.BALL_SIZE >= brick['x'] and
                self.ball_x <= brick['x'] + self.BRICK_W and
                self.ball_y + self.BALL_SIZE >= brick['y'] and
                self.ball_y <= brick['y'] + self.BRICK_H):
                brick['alive'] = False
                self.ball_vy = -self.ball_vy
                reward = 1
                self.score += 1
                break
        
        # Ball lost
        if self.ball_y >= self.HEIGHT:
            self.done = True
            reward = -1
        
        # Win condition
        if all(not b['alive'] for b in self.bricks):
            self.done = True
            reward = 10
        
        return self._render(), reward, self.done, {'score': self.score}
    
    def get_valid_actions(self):
        return [0, 1, 2]


class PixelInvaders:
    """
    Simple Space Invaders with pixel observations.
    
    Screen: 84x84
    Objects: Player, Enemies (grid), Bullets
    Actions: 0=stay, 1=left, 2=right, 3=shoot
    """
    
    WIDTH = 84
    HEIGHT = 84
    PLAYER_W = 6
    PLAYER_H = 4
    ENEMY_W = 5
    ENEMY_H = 3
    N_ENEMY_COLS = 8
    N_ENEMY_ROWS = 4
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.player_x = self.WIDTH // 2 - self.PLAYER_W // 2
        
        # Enemies
        self.enemies = []
        start_x = 5
        for row in range(self.N_ENEMY_ROWS):
            for col in range(self.N_ENEMY_COLS):
                self.enemies.append({
                    'x': start_x + col * (self.ENEMY_W + 2),
                    'y': 10 + row * (self.ENEMY_H + 2),
                    'alive': True
                })
        
        self.enemy_dir = 1
        self.enemy_move_timer = 0
        
        self.player_bullets = []
        self.enemy_bullets = []
        
        self.score = 0
        self.done = False
        
        return self._render()
    
    def _render(self) -> np.ndarray:
        screen = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.uint8)
        
        # Draw enemies
        for e in self.enemies:
            if e['alive']:
                x, y = int(e['x']), int(e['y'])
                screen[max(0,y):min(self.HEIGHT,y+self.ENEMY_H),
                       max(0,x):min(self.WIDTH,x+self.ENEMY_W)] = 180
        
        # Draw player
        px = int(self.player_x)
        screen[self.HEIGHT-10:self.HEIGHT-10+self.PLAYER_H,
               max(0,px):min(self.WIDTH,px+self.PLAYER_W)] = 255
        
        # Draw bullets
        for b in self.player_bullets:
            y = int(b['y'])
            x = int(b['x'])
            if 0 <= y < self.HEIGHT and 0 <= x < self.WIDTH:
                screen[max(0,y):min(self.HEIGHT,y+3), x:x+1] = 255
        
        for b in self.enemy_bullets:
            y = int(b['y'])
            x = int(b['x'])
            if 0 <= y < self.HEIGHT and 0 <= x < self.WIDTH:
                screen[max(0,y):min(self.HEIGHT,y+3), x:x+1] = 128
        
        return screen
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Move player
        if action == 1:
            self.player_x = max(0, self.player_x - 3)
        elif action == 2:
            self.player_x = min(self.WIDTH - self.PLAYER_W, self.player_x + 3)
        elif action == 3 and len(self.player_bullets) < 3:
            self.player_bullets.append({
                'x': self.player_x + self.PLAYER_W // 2,
                'y': self.HEIGHT - 12
            })
        
        # Move bullets
        self.player_bullets = [{'x': b['x'], 'y': b['y'] - 4} 
                               for b in self.player_bullets if b['y'] > 0]
        self.enemy_bullets = [{'x': b['x'], 'y': b['y'] + 3}
                              for b in self.enemy_bullets if b['y'] < self.HEIGHT]
        
        reward = 0
        
        # Move enemies
        self.enemy_move_timer += 1
        if self.enemy_move_timer >= 10:
            self.enemy_move_timer = 0
            
            # Check edges
            hit_edge = False
            for e in self.enemies:
                if e['alive']:
                    if e['x'] <= 2 or e['x'] >= self.WIDTH - self.ENEMY_W - 2:
                        hit_edge = True
                        break
            
            if hit_edge:
                self.enemy_dir = -self.enemy_dir
                for e in self.enemies:
                    e['y'] += 3
            else:
                for e in self.enemies:
                    e['x'] += self.enemy_dir * 2
            
            # Random enemy shoots
            alive_enemies = [e for e in self.enemies if e['alive']]
            if alive_enemies and self.rng.random() < 0.3:
                shooter = self.rng.choice(alive_enemies)
                self.enemy_bullets.append({
                    'x': shooter['x'] + self.ENEMY_W // 2,
                    'y': shooter['y'] + self.ENEMY_H
                })
        
        # Check bullet-enemy collisions
        for bullet in self.player_bullets[:]:
            for enemy in self.enemies:
                if not enemy['alive']:
                    continue
                if (bullet['x'] >= enemy['x'] and 
                    bullet['x'] <= enemy['x'] + self.ENEMY_W and
                    bullet['y'] >= enemy['y'] and 
                    bullet['y'] <= enemy['y'] + self.ENEMY_H):
                    enemy['alive'] = False
                    self.player_bullets.remove(bullet)
                    reward = 1
                    self.score += 1
                    break
        
        # Check bullet-player collision
        for bullet in self.enemy_bullets:
            if (bullet['x'] >= self.player_x and
                bullet['x'] <= self.player_x + self.PLAYER_W and
                bullet['y'] >= self.HEIGHT - 10):
                self.done = True
                reward = -10
        
        # Check enemy reached bottom
        for e in self.enemies:
            if e['alive'] and e['y'] >= self.HEIGHT - 15:
                self.done = True
                reward = -10
        
        # Win
        if all(not e['alive'] for e in self.enemies):
            self.done = True
            reward = 50
        
        return self._render(), reward, self.done, {'score': self.score}
    
    def get_valid_actions(self):
        return [0, 1, 2, 3]


# =============================================================================
# TOKENIZERS - Different ways to convert pixels to tokens
# =============================================================================

def tokenize_raw(screen: np.ndarray, threshold: int = 50) -> Set[str]:
    """
    RAW: Every bright pixel is a token.
    Warning: Huge state space!
    """
    tokens = set()
    bright = np.where(screen > threshold)
    for y, x in zip(bright[0], bright[1]):
        tokens.add(f"p_{y}_{x}")
    return tokens


def tokenize_grid(screen: np.ndarray, grid_size: int = 7, threshold: int = 50) -> Set[str]:
    """
    GRID: Divide into NxN regions, summarize each.
    Much smaller state space.
    """
    tokens = set()
    h, w = screen.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            region = screen[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
            
            # What's in this region?
            mean_val = region.mean()
            max_val = region.max()
            
            if max_val > 200:  # Bright object
                tokens.add(f"g_{gy}_{gx}_bright")
            elif max_val > 100:  # Medium
                tokens.add(f"g_{gy}_{gx}_med")
            elif max_val > threshold:  # Dim
                tokens.add(f"g_{gy}_{gx}_dim")
    
    return tokens


def tokenize_diff(screen: np.ndarray, prev_screen: np.ndarray, 
                  grid_size: int = 7, threshold: int = 30) -> Set[str]:
    """
    DIFF: Track what changed between frames.
    Good for detecting motion.
    """
    tokens = set()
    h, w = screen.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    
    diff = screen.astype(np.int16) - prev_screen.astype(np.int16)
    
    for gy in range(grid_size):
        for gx in range(grid_size):
            region = diff[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
            
            pos_change = (region > threshold).sum()
            neg_change = (region < -threshold).sum()
            
            if pos_change > 5:
                tokens.add(f"d_{gy}_{gx}_appear")
            if neg_change > 5:
                tokens.add(f"d_{gy}_{gx}_disappear")
    
    # Also include current state
    tokens.update(tokenize_grid(screen, grid_size, threshold))
    
    return tokens


def tokenize_objects(screen: np.ndarray, threshold: int = 100) -> Set[str]:
    """
    OBJECTS: Detect connected components as objects.
    Returns bounding boxes and centers.
    """
    from scipy import ndimage
    
    tokens = set()
    
    # Threshold and label
    binary = screen > threshold
    labeled, n_objects = ndimage.label(binary)
    
    for obj_id in range(1, n_objects + 1):
        # Find bounding box
        ys, xs = np.where(labeled == obj_id)
        if len(ys) == 0:
            continue
        
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        # Center (quantized to grid)
        cy = (y_min + y_max) // 2 // 12
        cx = (x_min + x_max) // 2 // 12
        
        # Size
        area = len(ys)
        if area < 10:
            size = "tiny"
        elif area < 50:
            size = "small"
        elif area < 200:
            size = "med"
        else:
            size = "large"
        
        tokens.add(f"obj_{cy}_{cx}_{size}")
    
    return tokens


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PIXEL ENVIRONMENTS TEST")
    print("="*60)
    
    # Test Pong
    print("\n--- PONG ---")
    env = PixelPong(seed=42)
    screen = env.reset()
    print(f"Screen shape: {screen.shape}")
    print(f"Screen dtype: {screen.dtype}")
    print(f"Pixel range: {screen.min()}-{screen.max()}")
    
    # Test tokenizers
    print(f"\nRAW tokens: {len(tokenize_raw(screen))}")
    print(f"GRID tokens (7x7): {len(tokenize_grid(screen, 7))}")
    print(f"GRID tokens (12x12): {len(tokenize_grid(screen, 12))}")
    
    # Test step
    screen2, reward, done, info = env.step(1)  # Move up
    diff_tokens = tokenize_diff(screen2, screen)
    print(f"DIFF tokens after action: {len(diff_tokens)}")
    
    try:
        obj_tokens = tokenize_objects(screen)
        print(f"OBJECT tokens: {len(obj_tokens)}")
    except ImportError:
        print("scipy not available for object detection")
    
    # Test Breakout
    print("\n--- BREAKOUT ---")
    env = PixelBreakout(seed=42)
    screen = env.reset()
    print(f"GRID tokens: {len(tokenize_grid(screen, 7))}")
    
    # Test Invaders
    print("\n--- INVADERS ---")
    env = PixelInvaders(seed=42)
    screen = env.reset()
    print(f"GRID tokens: {len(tokenize_grid(screen, 7))}")
    
    print("\n" + "="*60)
    print("Environments ready for rule learning!")
