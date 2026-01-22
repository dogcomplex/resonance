"""
MiniRogue: A roguelike environment with probabilistic transitions

Features:
- 6x6 procedural dungeon
- 5 enemy types with different stats
- Weapons and armor
- Potions and status effects (poison)
- Random encounters, miss chances, variable damage
- Tests probabilistic rule learning
"""

import random
from typing import Set, FrozenSet, List, Tuple, Dict

class MiniRogue:
    """Roguelike with probabilistic combat."""
    
    ENEMIES = {
        "rat": {"hp": 3, "dmg": 1, "def": 0, "acc": 0.9},
        "goblin": {"hp": 8, "dmg": 3, "def": 1, "acc": 0.8},
        "skeleton": {"hp": 12, "dmg": 4, "def": 2, "acc": 0.85},
        "spider": {"hp": 6, "dmg": 2, "def": 0, "acc": 0.95, "special": "poison"},
        "ogre": {"hp": 20, "dmg": 6, "def": 3, "acc": 0.7},
    }
    
    WEAPONS = {"dagger": 2, "sword": 4, "axe": 6}
    ARMORS = {"leather": 1, "chainmail": 3}
    
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.reset()
    
    def _generate_dungeon(self):
        """Generate 6x6 procedural dungeon."""
        self.rooms = {}
        for x in range(6):
            for y in range(6):
                roll = self.rng.random()
                if (x, y) == (0, 0):
                    room = {"type": "start", "enemies": [], "items": []}
                elif (x, y) == (5, 5):
                    room = {"type": "exit", "enemies": ["ogre"], "items": []}
                elif roll < 0.4:
                    pool = ["rat", "goblin"] if x + y < 5 else ["skeleton", "spider"]
                    room = {"type": "monster", "enemies": [self.rng.choice(pool)], "items": []}
                elif roll < 0.55:
                    room = {"type": "treasure", "enemies": [], 
                           "items": [self.rng.choice(["potion", "dagger", "leather"])]}
                else:
                    room = {"type": "empty", "enemies": [], "items": []}
                self.rooms[(x, y)] = room
    
    def reset(self):
        self._generate_dungeon()
        self.x, self.y = 0, 0
        self.hp, self.max_hp = 25, 25
        self.weapon, self.armor = None, None
        self.inventory: List[str] = []
        self.poisoned = 0
        self.in_combat, self.enemy, self.enemy_hp = False, None, 0
        self.defeated: Set[Tuple] = set()
        self.collected: Set[Tuple] = set()
    
    def to_tokens(self) -> FrozenSet[str]:
        tokens = set()
        room = self.rooms[(self.x, self.y)]
        
        tokens.add(f"at_{self.x}_{self.y}")
        tokens.add(f"room_{room['type']}")
        
        hp_pct = self.hp / self.max_hp
        if hp_pct <= 0: tokens.add("dead")
        elif hp_pct <= 0.3: tokens.add("hp_critical")
        elif hp_pct <= 0.6: tokens.add("hp_low")
        else: tokens.add("hp_ok")
        
        if self.weapon:
            tokens.add(f"weapon_{self.weapon}")
            tokens.add("armed")
        else:
            tokens.add("unarmed")
        
        if self.armor:
            tokens.add(f"armor_{self.armor}")
        
        for item in set(self.inventory):
            tokens.add(f"has_{item}")
        
        if self.poisoned:
            tokens.add("poisoned")
        
        for enemy in room["enemies"]:
            if (self.x, self.y, enemy) not in self.defeated:
                tokens.add(f"see_{enemy}")
        
        for item in room["items"]:
            if (self.x, self.y, item) not in self.collected:
                tokens.add(f"see_{item}")
        
        if self.in_combat:
            tokens.add("in_combat")
            tokens.add(f"fighting_{self.enemy}")
        
        room = self.rooms[(self.x, self.y)]
        if self.x == 5 and self.y == 5:
            if not any((5, 5, e) not in self.defeated for e in room["enemies"]):
                tokens.add("won")
        
        return frozenset(tokens)
    
    def valid_actions(self) -> List[str]:
        if self.hp <= 0:
            return []
        
        room = self.rooms[(self.x, self.y)]
        actions = []
        
        if self.in_combat:
            actions.extend(["attack", "flee"])
            if "potion" in self.inventory:
                actions.append("use_potion")
            return actions
        
        for dx, dy, d in [(-1,0,"west"), (1,0,"east"), (0,-1,"north"), (0,1,"south")]:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < 6 and 0 <= ny < 6:
                actions.append(f"move_{d}")
        
        for enemy in room["enemies"]:
            if (self.x, self.y, enemy) not in self.defeated:
                actions.append(f"fight_{enemy}")
        
        for item in room["items"]:
            if (self.x, self.y, item) not in self.collected:
                actions.append(f"pickup_{item}")
        
        if "potion" in self.inventory and self.hp < self.max_hp:
            actions.append("use_potion")
        
        for item in self.inventory:
            if item in self.WEAPONS and self.weapon != item:
                actions.append(f"equip_{item}")
            if item in self.ARMORS and self.armor != item:
                actions.append(f"equip_{item}")
        
        return actions
    
    def step(self, action: str) -> Tuple[FrozenSet[str], str, bool]:
        outcome, done = "ok", False
        
        # Poison damage
        if self.poisoned > 0:
            self.hp -= 1
            self.poisoned -= 1
        
        room = self.rooms[(self.x, self.y)]
        
        if self.in_combat:
            e = self.ENEMIES[self.enemy]
            
            if action == "attack":
                dmg = self.WEAPONS.get(self.weapon, 1)
                if self.rng.random() < 0.85:  # 85% hit rate
                    actual = max(1, dmg - e["def"])
                    self.enemy_hp -= actual
                    if self.enemy_hp <= 0:
                        self.defeated.add((self.x, self.y, self.enemy))
                        self.in_combat = False
                        outcome = f"killed_{self.enemy}"
                    else:
                        outcome = f"hit_{self.enemy}"
                else:
                    outcome = "missed"
                
                # Enemy counterattack
                if self.in_combat and self.rng.random() < e["acc"]:
                    defense = self.ARMORS.get(self.armor, 0)
                    taken = max(1, e["dmg"] - defense)
                    self.hp -= taken
                    outcome += f"_took_{taken}"
                    
                    if e.get("special") == "poison" and self.rng.random() < 0.3:
                        self.poisoned = 3
                        outcome += "_poisoned"
            
            elif action == "flee":
                if self.rng.random() < 0.7:  # 70% flee success
                    self.in_combat = False
                    outcome = "fled"
                else:
                    outcome = "flee_failed"
            
            elif action == "use_potion":
                self.inventory.remove("potion")
                self.hp = min(self.max_hp, self.hp + 8)
                outcome = "healed"
        
        elif action.startswith("move_"):
            d = action[5:]
            dx, dy = {"west":(-1,0), "east":(1,0), "north":(0,-1), "south":(0,1)}[d]
            self.x, self.y = self.x + dx, self.y + dy
            new_room = self.rooms[(self.x, self.y)]
            outcome = f"entered_{new_room['type']}"
            
            # Random encounter (25%)
            available = [e for e in new_room["enemies"] 
                        if (self.x, self.y, e) not in self.defeated]
            if available and self.rng.random() < 0.25:
                self.enemy = self.rng.choice(available)
                self.enemy_hp = self.ENEMIES[self.enemy]["hp"]
                self.in_combat = True
                outcome += "_ambush"
        
        elif action.startswith("fight_"):
            self.enemy = action[6:]
            self.enemy_hp = self.ENEMIES[self.enemy]["hp"]
            self.in_combat = True
            outcome = f"engaged_{self.enemy}"
        
        elif action.startswith("pickup_"):
            item = action[7:]
            self.collected.add((self.x, self.y, item))
            self.inventory.append(item)
            outcome = f"got_{item}"
        
        elif action.startswith("equip_"):
            item = action[6:]
            if item in self.WEAPONS:
                if self.weapon:
                    self.inventory.append(self.weapon)
                self.inventory.remove(item)
                self.weapon = item
            elif item in self.ARMORS:
                if self.armor:
                    self.inventory.append(self.armor)
                self.inventory.remove(item)
                self.armor = item
            outcome = f"equipped_{item}"
        
        elif action == "use_potion":
            self.inventory.remove("potion")
            self.hp = min(self.max_hp, self.hp + 8)
            outcome = "healed"
        
        if self.hp <= 0:
            outcome, done = "died", True
        
        room = self.rooms[(self.x, self.y)]
        if self.x == 5 and self.y == 5:
            if not any((5, 5, e) not in self.defeated for e in room["enemies"]):
                outcome, done = "won", True
        
        return self.to_tokens(), outcome, done


def action_to_tokens(action: str) -> FrozenSet[str]:
    return frozenset({f"action_{action}"})
