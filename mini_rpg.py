"""
Mini RPG - Test environment for token-based learning

A simple RPG with:
- 3x3 grid world
- Combat with different enemies
- Items (sword, crown)
- Conditional mechanics (sword affects combat)
"""

import random
from typing import Set, FrozenSet, List, Tuple

class MiniRPG:
    """Simple RPG for testing token-based state learning."""
    
    ROOMS = {
        (0,0): {"name": "village", "enemies": []},
        (1,0): {"name": "forest", "enemies": ["rat"]},
        (2,0): {"name": "cave", "enemies": ["goblin"]},
        (0,1): {"name": "path", "enemies": []},
        (1,1): {"name": "clearing", "enemies": ["rat", "goblin"]},
        (2,1): {"name": "dungeon", "enemies": ["boss"]},
        (0,2): {"name": "shop", "enemies": [], "shop": True},
        (1,2): {"name": "armory", "enemies": [], "items": ["sword"]},
        (2,2): {"name": "treasure", "enemies": [], "items": ["crown"]},
    }
    
    ENEMY_HP = {"rat": 2, "goblin": 5, "boss": 10}
    ENEMY_DMG = {"rat": 1, "goblin": 2, "boss": 4}
    
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self):
        self.x, self.y = 0, 0
        self.hp = 10
        self.has_sword = False
        self.has_crown = False
        self.gold = 5
        self.defeated: Set[Tuple] = set()
        self.collected: Set[Tuple] = set()
        self.in_combat = False
        self.enemy = None
    
    def to_tokens(self) -> FrozenSet[str]:
        """Convert game state to observation tokens."""
        tokens = set()
        room = self.ROOMS[(self.x, self.y)]
        
        # Location
        tokens.add(f"at_{self.x}_{self.y}")
        tokens.add(f"room_{room['name']}")
        
        # HP
        tokens.add(f"hp_{self.hp}")
        if self.hp <= 3: tokens.add("hp_low")
        if self.hp >= 8: tokens.add("hp_high")
        
        # Equipment
        if self.has_sword: 
            tokens.add("has_sword")
        else:
            tokens.add("no_sword")
        
        if self.has_crown:
            tokens.add("has_crown")
        
        # Room contents
        for enemy in room.get("enemies", []):
            if (self.x, self.y, enemy) not in self.defeated:
                tokens.add(f"see_{enemy}")
        
        for item in room.get("items", []):
            if (self.x, self.y, item) not in self.collected:
                tokens.add(f"see_{item}")
        
        if room.get("shop"):
            tokens.add("at_shop")
        
        # Combat state
        if self.in_combat:
            tokens.add("in_combat")
            tokens.add(f"fighting_{self.enemy}")
        
        # Terminal states
        if self.hp <= 0:
            tokens.add("dead")
        if self.has_crown and self.x == 0 and self.y == 0:
            tokens.add("won")
        
        return frozenset(tokens)
    
    def valid_actions(self) -> List[str]:
        """Get valid actions in current state."""
        if self.hp <= 0:
            return []
        
        if self.in_combat:
            return ["attack", "flee"]
        
        actions = []
        room = self.ROOMS[(self.x, self.y)]
        
        # Movement
        for dx, dy, d in [(-1,0,"left"), (1,0,"right"), (0,-1,"up"), (0,1,"down")]:
            if (self.x + dx, self.y + dy) in self.ROOMS:
                actions.append(f"move_{d}")
        
        # Combat initiation
        for enemy in room.get("enemies", []):
            if (self.x, self.y, enemy) not in self.defeated:
                actions.append(f"fight_{enemy}")
        
        # Item pickup
        for item in room.get("items", []):
            if (self.x, self.y, item) not in self.collected:
                actions.append(f"pickup_{item}")
        
        return actions
    
    def step(self, action: str) -> Tuple[FrozenSet[str], str, bool]:
        """
        Execute action.
        
        Returns: (new_tokens, outcome, done)
        """
        outcome = "ok"
        done = False
        
        if self.in_combat:
            if action == "attack":
                damage = 3 if self.has_sword else 1
                enemy_hp = self.ENEMY_HP[self.enemy]
                
                if damage >= enemy_hp:
                    self.defeated.add((self.x, self.y, self.enemy))
                    self.in_combat = False
                    outcome = f"killed_{self.enemy}"
                    self.gold += 2
                else:
                    self.hp -= self.ENEMY_DMG[self.enemy]
                    outcome = "traded_blows"
                    
            elif action == "flee":
                self.in_combat = False
                self.hp -= 1
                outcome = "fled"
        
        elif action.startswith("move_"):
            direction = action[5:]
            dx, dy = {"left":(-1,0), "right":(1,0), "up":(0,-1), "down":(0,1)}[direction]
            self.x += dx
            self.y += dy
            room = self.ROOMS[(self.x, self.y)]
            outcome = f"entered_{room['name']}"
            
            # Random encounter
            available = [e for e in room.get("enemies", []) 
                        if (self.x, self.y, e) not in self.defeated]
            if available and self.rng.random() < 0.3:
                self.enemy = self.rng.choice(available)
                self.in_combat = True
                outcome += f"_ambush_{self.enemy}"
        
        elif action.startswith("fight_"):
            self.enemy = action[6:]
            self.in_combat = True
            outcome = f"engaged_{self.enemy}"
        
        elif action.startswith("pickup_"):
            item = action[7:]
            self.collected.add((self.x, self.y, item))
            if item == "sword":
                self.has_sword = True
            if item == "crown":
                self.has_crown = True
            outcome = f"got_{item}"
        
        # Check terminal states
        if self.hp <= 0:
            outcome = "died"
            done = True
        
        if self.has_crown and self.x == 0 and self.y == 0:
            done = True
            outcome = "won"
        
        return self.to_tokens(), outcome, done


def action_to_tokens(action: str) -> FrozenSet[str]:
    """Convert action string to token set."""
    return frozenset({f"action_{action}"})
