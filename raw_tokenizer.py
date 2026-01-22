"""
RAW TOKENIZER - No domain knowledge, no cheating.

Only uses:
- Raw numeric values from observation grid
- Position relative to agent
- No semantic labels (door, key, goal)
- No computed features (goal_distance, blocked, need_key)

The learner must discover all semantics from experience.
"""

from typing import FrozenSet, Tuple
import numpy as np


def raw_tokenize(obs: np.ndarray, carrying: Tuple[int, int, int] = None) -> FrozenSet[str]:
    """
    Convert MiniGrid observation to raw tokens.
    
    No domain knowledge - just encodes what we see.
    
    Observation format: 7x7x3 grid
    - obs[r, c, 0] = object type (integer)
    - obs[r, c, 1] = color (integer)
    - obs[r, c, 2] = state (integer)
    
    Agent is at row 6, col 3, facing up (toward row 0).
    """
    tokens = set()
    
    # What's directly in front (one cell ahead)
    front = obs[5, 3]
    tokens.add(f"front_t{int(front[0])}")  # type
    if front[0] not in [0, 1, 2, 3]:  # Not unseen/empty/wall/floor
        tokens.add(f"front_c{int(front[1])}")  # color (only if meaningful)
        tokens.add(f"front_s{int(front[2])}")  # state
    
    # What's to the left (agent's left = col 2 at row 5)
    left = obs[5, 2]
    tokens.add(f"left_t{int(left[0])}")
    
    # What's to the right (agent's right = col 4 at row 5)
    right = obs[5, 4]
    tokens.add(f"right_t{int(right[0])}")
    
    # What we're carrying (if anything)
    if carrying and carrying[0] != 0:
        tokens.add(f"carry_t{int(carrying[0])}")
        tokens.add(f"carry_c{int(carrying[1])}")
    else:
        tokens.add("carry_none")
    
    # Scan visible area for notable objects (types 4-9)
    # We don't name them - just note their presence and rough location
    for r in range(7):
        for c in range(7):
            obj_type = int(obs[r, c, 0])
            if obj_type >= 4:  # Something interesting (not empty/wall/floor)
                # Relative position: negative = ahead, positive = behind
                rel_row = r - 6  # -6 to 0 (agent at row 6)
                rel_col = c - 3  # -3 to 3 (agent at col 3)
                
                # Coarse distance bucket
                dist = abs(rel_row) + abs(rel_col)
                if dist <= 1:
                    loc = "adj"
                elif dist <= 3:
                    loc = "near"
                else:
                    loc = "far"
                
                tokens.add(f"see_t{obj_type}_{loc}")
                
                # For doors (type 4), include state
                if obj_type == 4:
                    tokens.add(f"door_s{int(obs[r, c, 2])}")
    
    return frozenset(tokens)


def test_raw_tokenizer():
    """Test the raw tokenizer."""
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from minigrid_official import EmptyEnv, DoorKeyEnv
    
    print("=" * 60)
    print("RAW TOKENIZER TEST - No Domain Knowledge")
    print("=" * 60)
    
    print("\n1. Empty-5x5:")
    env = EmptyEnv(size=5)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    carrying = env.carrying.encode() if env.carrying else None
    
    tokens = raw_tokenize(obs, carrying)
    print(f"   Tokens: {sorted(tokens)}")
    
    # Show what the types mean (but learner doesn't know this!)
    print("\n   [Reference - learner doesn't see this]")
    print("   t1=empty, t2=wall, t4=door, t5=key, t8=goal, t9=lava")
    
    print("\n2. DoorKey-5x5:")
    env = DoorKeyEnv(size=5)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    carrying = env.carrying.encode() if env.carrying else None
    
    tokens = raw_tokenize(obs, carrying)
    print(f"   Tokens: {sorted(tokens)}")
    
    # Take some actions
    print("\n   After picking up key (if in front):")
    for _ in range(10):
        obs, _, term, _, _ = env.step(2)  # forward
        if term:
            break
    obs, _, _, _, _ = env.step(3)  # pickup
    
    carrying = env.carrying.encode() if env.carrying else None
    tokens = raw_tokenize(obs, carrying)
    print(f"   Tokens: {sorted(tokens)}")


if __name__ == "__main__":
    test_raw_tokenizer()
