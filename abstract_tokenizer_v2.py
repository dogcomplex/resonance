"""
Abstract tokenizer V2 - includes directional context.

Key insight: For rules about turning/movement, we need to know
what's to the left and right, not just what's in front.
"""

from typing import FrozenSet, Tuple
import numpy as np

# Object types
UNSEEN, EMPTY, WALL, FLOOR, DOOR, KEY, BALL, BOX, GOAL, LAVA, AGENT = range(11)
OBJ_NAMES = {0: 'unseen', 1: 'empty', 2: 'wall', 3: 'floor', 4: 'door', 
             5: 'key', 6: 'ball', 7: 'box', 8: 'goal', 9: 'lava', 10: 'agent'}
DOOR_OPEN, DOOR_CLOSED, DOOR_LOCKED = 0, 1, 2


def abstract_tokenize_v2(obs: np.ndarray, carrying: Tuple[int, int, int] = None) -> FrozenSet[str]:
    """
    Create tokens including directional context.
    
    Agent view (7x7 grid):
    - Agent at row 6, col 3
    - Forward = row 5
    - Left = col 2
    - Right = col 4
    """
    tokens = set()
    
    # What's directly in front (row 5, col 3)
    front_type = int(obs[5, 3, 0])
    tokens.add(f"front_{OBJ_NAMES.get(front_type, front_type)}")
    
    # Door state if facing door
    if front_type == DOOR:
        state = int(obs[5, 3, 2])
        states = ['open', 'closed', 'locked']
        tokens.add(f"front_door_{states[state]}")
    
    # What's to the left (at agent's position, one column left = col 2)
    # Actually, "left" means what we'd face after turning left
    # When facing up (forward = -row), left = -col, right = +col
    # At row 6, the cell to the left of forward view is col 2, row 5
    left_type = int(obs[5, 2, 0])
    tokens.add(f"left_{OBJ_NAMES.get(left_type, left_type)}")
    
    # What's to the right (col 4, row 5)
    right_type = int(obs[5, 4, 0])
    tokens.add(f"right_{OBJ_NAMES.get(right_type, right_type)}")
    
    # Carrying
    if carrying:
        tokens.add(f"carrying_{OBJ_NAMES.get(carrying[0], carrying[0])}")
    else:
        tokens.add("empty_handed")
    
    # Important objects in view (presence only)
    for r in range(7):
        for c in range(7):
            obj_type = int(obs[r, c, 0])
            if obj_type in [KEY, GOAL, DOOR]:
                tokens.add(f"see_{OBJ_NAMES[obj_type]}")
                if obj_type == DOOR:
                    state = int(obs[r, c, 2])
                    tokens.add(f"door_{'open' if state == 0 else 'closed' if state == 1 else 'locked'}")
    
    # Goal distance
    for r in range(7):
        for c in range(7):
            if int(obs[r, c, 0]) == GOAL:
                dist = abs(6 - r) + abs(3 - c)
                if dist <= 1:
                    tokens.add("goal_adjacent")
                elif dist <= 3:
                    tokens.add("goal_near")
                else:
                    tokens.add("goal_far")
    
    # Blocked?
    if front_type == WALL:
        tokens.add("blocked")
    elif front_type == DOOR and int(obs[5, 3, 2]) != DOOR_OPEN:
        tokens.add("blocked")
        if int(obs[5, 3, 2]) == DOOR_LOCKED:
            tokens.add("need_key")
    
    return frozenset(tokens)


# Test
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from minigrid_official import EmptyEnv
    
    print("Testing tokenizer V2:")
    env = EmptyEnv(size=5)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    
    carrying = env.carrying.encode() if env.carrying else None
    tokens = abstract_tokenize_v2(obs, carrying)
    print(f"  Initial: {sorted(tokens)}")
    
    # Turn right
    obs, _, _, _, _ = env.step(1)  # right
    carrying = env.carrying.encode() if env.carrying else None
    tokens = abstract_tokenize_v2(obs, carrying)
    print(f"  After right: {sorted(tokens)}")
    
    # Turn left twice
    for _ in range(2):
        obs, _, _, _, _ = env.step(0)  # left
    carrying = env.carrying.encode() if env.carrying else None
    tokens = abstract_tokenize_v2(obs, carrying)
    print(f"  After 2 lefts: {sorted(tokens)}")
