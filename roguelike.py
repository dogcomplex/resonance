"""
Roguelike Dungeon - Test environment for token-based learning

Features:
- 8x8 grid
- 4 enemy types with different stats
- Weapons and armor (affect combat)
- Potions (healing)
- Stochastic elements (flee success, ambushes)
"""

from collections import defaultdict
import random
from typing import FrozenSet, List, Dict, Tuple

ITEMS = {
    "dagger": ("weapon", 2), "sword": ("weapon", 4),
    "leather": ("armor", 1), "chain": ("armor", 2),
    "heal_potion": ("potion", 10),
}

ENEMIES = {
    # name: (hp, damage, defense)
    "rat": (5, 1, 0),
    "goblin": (10, 3, 1),
    "skeleton": (15, 4, 2),
    "ogre": (30, 6, 3),
}


class Roguelike:
    SIZE = 8
    
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.reset()
    
    def reset(self):
        self.px, self.py = 2, 2
        self.hp, self.max_hp = 30, 30
        self.weapon = None
        self.armor = None
        self.inventory: Dict[str, int] = defaultdict(int)
        self.in_combat = False
        self.enemy = None
        self.enemy_hp = 0
        self.floor = 1
        
        # Generate dungeon
        self.enemies: Dict[Tuple[int, int], str] = {}
        self.items: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        
        for _ in range(3):
            x, y = self.rng.randint(1, 6), self.rng.randint(1, 6)
            if (x, y) != (2, 2):
                self.enemies[(x, y)] = self.rng.choice(list(ENEMIES.keys()))
        
        for _ in range(2):
            x, y = self.rng.randint(1, 6), self.rng.randint(1, 6)
            self.items[(x, y)].append(self.rng.choice(list(ITEMS.keys())))
    
    def to_tokens(self) -> FrozenSet[str]:
        t = set()
        
        # Position
        t.add(f"pos_{self.px}_{self.py}")
        t.add(f"region_{'left' if self.px < 4 else 'right'}_{'top' if self.py < 4 else 'bottom'}")
        
        # HP
        hp_pct = self.hp / self.max_hp
        if hp_pct <= 0: t.add("dead")
        elif hp_pct <= 0.3: t.add("hp_low")
        elif hp_pct <= 0.7: t.add("hp_medium")
        else: t.add("hp_high")
        
        # Equipment
        if self.weapon:
            t.add(f"wielding_{self.weapon}")
            t.add("armed")
        else:
            t.add("unarmed")
        
        if self.armor:
            t.add(f"wearing_{self.armor}")
            t.add("armored")
        else:
            t.add("unarmored")
        
        # Inventory
        if self.inventory.get("heal_potion", 0):
            t.add("has_potions")
        
        # Combat
        if self.in_combat:
            t.add("in_combat")
            t.add(f"fighting_{self.enemy}")
            ehp_pct = self.enemy_hp / ENEMIES[self.enemy][0]
            if ehp_pct <= 0.3: t.add("enemy_low")
            elif ehp_pct <= 0.7: t.add("enemy_medium")
            else: t.add("enemy_high")
        
        # Visible entities
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = self.px + dx, self.py + dy
                if 0 <= nx < self.SIZE and 0 <= ny < self.SIZE:
                    if (nx, ny) in self.enemies:
                        e = self.enemies[(nx, ny)]
                        if abs(dx) <= 1 and abs(dy) <= 1:
                            t.add(f"adjacent_{e}")
                        else:
                            t.add(f"nearby_{e}")
                    if dx == 0 and dy == 0:
                        for item in self.items.get((nx, ny), []):
                            t.add(f"see_{item}")
        
        return frozenset(t)
    
    def valid_actions(self) -> List[str]:
        if self.hp <= 0:
            return []
        
        if self.in_combat:
            acts = ["attack", "flee"]
            if self.inventory.get("heal_potion", 0):
                acts.append("use_heal_potion")
            return acts
        
        acts = []
        for dx, dy, d in [(0,-1,"up"), (0,1,"down"), (-1,0,"left"), (1,0,"right")]:
            nx, ny = self.px + dx, self.py + dy
            if 0 < nx < self.SIZE-1 and 0 < ny < self.SIZE-1:
                acts.append(f"move_{d}")
        
        if self.items.get((self.px, self.py)):
            acts.append("pickup")
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                pos = (self.px + dx, self.py + dy)
                if pos in self.enemies:
                    acts.append(f"fight_{self.enemies[pos]}")
        
        if self.inventory.get("heal_potion", 0) and self.hp < self.max_hp:
            acts.append("use_heal_potion")
        
        return list(set(acts))
    
    def step(self, action: str) -> Tuple[FrozenSet[str], str, bool]:
        outcome, done = "ok", False
        
        if self.in_combat:
            ehp, edmg, edef = ENEMIES[self.enemy]
            
            if action == "attack":
                dmg = 1 if not self.weapon else ITEMS[self.weapon][1]
                dmg = max(1, dmg - edef)
                self.enemy_hp -= dmg
                
                if self.enemy_hp <= 0:
                    self.in_combat = False
                    outcome = f"killed_{self.enemy}"
                else:
                    taken = edmg
                    if self.armor:
                        taken = max(1, taken - ITEMS[self.armor][1])
                    self.hp -= taken
                    outcome = "traded_blows"
            
            elif action == "flee":
                if self.rng.random() < 0.7:
                    self.in_combat = False
                    outcome = "fled"
                else:
                    taken = edmg
                    if self.armor:
                        taken = max(1, taken - ITEMS[self.armor][1])
                    self.hp -= taken
                    outcome = "flee_failed"
            
            elif action == "use_heal_potion":
                self.inventory["heal_potion"] -= 1
                self.hp = min(self.max_hp, self.hp + 10)
                outcome = "healed"
        
        elif action.startswith("move_"):
            d = action[5:]
            dx, dy = {"up":(0,-1), "down":(0,1), "left":(-1,0), "right":(1,0)}[d]
            nx, ny = self.px + dx, self.py + dy
            
            if (nx, ny) in self.enemies:
                self.in_combat = True
                self.enemy = self.enemies[(nx, ny)]
                self.enemy_hp = ENEMIES[self.enemy][0]
                del self.enemies[(nx, ny)]
                self.px, self.py = nx, ny
                outcome = f"ambushed_{self.enemy}"
            else:
                self.px, self.py = nx, ny
                outcome = "moved"
        
        elif action.startswith("fight_"):
            en = action[6:]
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    pos = (self.px + dx, self.py + dy)
                    if self.enemies.get(pos) == en:
                        self.in_combat = True
                        self.enemy = en
                        self.enemy_hp = ENEMIES[en][0]
                        del self.enemies[pos]
                        outcome = f"engaged_{en}"
                        break
        
        elif action == "pickup":
            for item in self.items[(self.px, self.py)]:
                itype, ipower = ITEMS[item]
                if itype == "weapon":
                    if not self.weapon or ITEMS[self.weapon][1] < ipower:
                        self.weapon = item
                elif itype == "armor":
                    if not self.armor or ITEMS[self.armor][1] < ipower:
                        self.armor = item
                else:
                    self.inventory[item] += 1
            self.items[(self.px, self.py)] = []
            outcome = "picked_up"
        
        elif action == "use_heal_potion":
            self.inventory["heal_potion"] -= 1
            self.hp = min(self.max_hp, self.hp + 10)
            outcome = "healed"
        
        if self.hp <= 0:
            done = True
        
        return self.to_tokens(), outcome, done


def action_to_tokens(action: str) -> FrozenSet[str]:
    return frozenset({f"action_{action}"})
