"""
Farm Game v3 - Fixed consume-all (X) logic.

Key fix: TokenX means "consume all of this token" but does NOT require
the token to be present. Having 0 is valid - you just consume 0.
"""

import re
import random
from typing import Set, List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Token:
    base: str
    quantity: int = 1
    consume_all: bool = False  # X suffix - consume all, but 0 is valid!
    is_range: bool = False
    range_low: int = 0
    range_high: int = 0


@dataclass
class Outcome:
    tokens: List[Token]
    probability: float


@dataclass
class Rule:
    id: str
    priority: int
    lhs: List[Token]
    rhs: List[Outcome]


def parse_recipes(filepath: str) -> List[Rule]:
    rules = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('ID,') or '=>' not in line:
                continue
            
            parts = line.split(',', 1)
            rule_id = parts[0].strip()
            recipe = parts[1].strip()
            
            lhs_str, rhs_str = recipe.split('=>', 1)
            
            # Parse priority (❗ or ❗N)
            priority = 0
            for m in re.finditer(r'❗(\d*)', lhs_str):
                p = m.group(1)
                priority = max(priority, int(p) if p else 1)
            lhs_str = re.sub(r'❗\d*', '', lhs_str)
            
            lhs = parse_tokens(lhs_str, is_lhs=True)
            rhs = parse_rhs(rhs_str)
            
            rules.append(Rule(rule_id, priority, lhs, rhs))
    
    return rules


def parse_tokens(s: str, is_lhs: bool = False) -> List[Token]:
    tokens = []
    for part in s.split():
        if not part:
            continue
        
        consume_all = 'X' in part and is_lhs
        part = part.replace('X', '')
        
        # Check for range: base+low-high
        range_match = re.match(r'(.+?)(\d+)-(\d+)$', part)
        if range_match:
            tokens.append(Token(
                base=range_match.group(1),
                is_range=True,
                range_low=int(range_match.group(2)),
                range_high=int(range_match.group(3))
            ))
            continue
        
        # Regular: base or base+qty
        match = re.match(r'(.+?)(\d+)?$', part)
        if match and match.group(1):
            base = match.group(1)
            qty = int(match.group(2)) if match.group(2) else 1
            tokens.append(Token(base=base, quantity=qty, consume_all=consume_all))
    
    return tokens


def parse_rhs(s: str) -> List[Outcome]:
    s = s.strip()
    outcomes = []
    
    if '%' in s:
        # Probabilistic - split by % markers
        for part in s.split():
            if '%' in part:
                idx = part.index('%')
                token_str = part[:idx]
                prob_str = part[idx+1:]
                prob = float(prob_str) if prob_str else 0
                tokens = parse_tokens(token_str)
                if tokens:
                    outcomes.append(Outcome(tokens, prob))
    else:
        tokens = parse_tokens(s)
        if tokens:
            outcomes.append(Outcome(tokens, 100))
    
    return outcomes


class FarmGame:
    def __init__(self, rules: List[Rule], seed: int = 42):
        self.rules = rules
        self.rng = random.Random(seed)
        self.state: Dict[str, int] = defaultdict(int)
    
    def reset(self, seed: int = None) -> Dict[str, int]:
        if seed is not None:
            self.rng = random.Random(seed)
        
        self.state = defaultdict(int)
        
        # Apply start rule
        self._apply_rule_by_id("ID_Start", force=True)
        
        # Spawn grid (priority 1-3 rules)
        self._run_auto_rules(min_priority=1, max_priority=10)
        
        # Give initial energy and day state
        self.state['💧'] = 225
        self.state['🌞'] = 9999
        
        return self._clean_state()
    
    def _clean_state(self) -> Dict[str, int]:
        return {k: v for k, v in self.state.items() if v > 0}
    
    def _apply_rule_by_id(self, rule_id: str, force: bool = False):
        for rule in self.rules:
            if rule.id == rule_id:
                self._apply_rule(rule, force)
                return True
        return False
    
    def _run_auto_rules(self, min_priority: int = 1, max_priority: int = 100, max_iter: int = 2000):
        """Run automatic high-priority rules until stable"""
        for _ in range(max_iter):
            applied = False
            for rule in sorted(self.rules, key=lambda r: -r.priority):
                if rule.priority < min_priority or rule.priority > max_priority:
                    continue
                if self._can_apply(rule):
                    self._apply_rule(rule)
                    applied = True
                    break
            if not applied:
                break
    
    def _can_apply(self, rule: Rule) -> bool:
        """
        Check if rule can be applied.
        FIXED: consume_all (X) tokens don't require presence - 0 is valid!
        """
        for token in rule.lhs:
            count = self.state.get(token.base, 0)
            if token.consume_all:
                # X means "consume all" - having 0 is fine!
                pass
            else:
                # Regular token - must have at least the required quantity
                if count < token.quantity:
                    return False
        return True
    
    def _apply_rule(self, rule: Rule, force: bool = False) -> bool:
        if not force and not self._can_apply(rule):
            return False
        
        # Consume LHS
        for token in rule.lhs:
            if token.consume_all:
                self.state[token.base] = 0
            else:
                self.state[token.base] -= token.quantity
        
        # Produce RHS
        if rule.rhs:
            total_prob = sum(o.probability for o in rule.rhs)
            
            if len(rule.rhs) == 1 and rule.rhs[0].probability >= 100:
                # Deterministic
                self._produce(rule.rhs[0].tokens)
            else:
                # Probabilistic
                roll = self.rng.random() * 100
                if roll < total_prob:
                    cumulative = 0
                    for outcome in rule.rhs:
                        cumulative += outcome.probability
                        if roll < cumulative:
                            self._produce(outcome.tokens)
                            break
                # else: no output (remaining probability)
        
        return True
    
    def _produce(self, tokens: List[Token]):
        for token in tokens:
            if token.is_range:
                qty = self.rng.randint(token.range_low, token.range_high)
            else:
                qty = token.quantity
            self.state[token.base] += qty
    
    def get_valid_actions(self) -> List[str]:
        """Get user-triggerable actions (exclude very high priority auto rules)"""
        valid = []
        for rule in self.rules:
            if rule.priority >= 20:  # Skip auto rules
                continue
            if self._can_apply(rule):
                valid.append(rule.id)
        return valid
    
    def step(self, rule_id: str) -> Tuple[Dict[str, int], bool]:
        """Apply a user action, then run automatic follow-up rules"""
        for rule in self.rules:
            if rule.id == rule_id:
                if self._can_apply(rule):
                    self._apply_rule(rule)
                    # Run automatic follow-ups (fish tick, etc.)
                    self._run_auto_rules(min_priority=5)
                    return self._clean_state(), True
                return self._clean_state(), False
        return self._clean_state(), False


class AnonymizedFarmGame:
    """Wrapper with full anonymization"""
    
    def __init__(self, rules: List[Rule], seed: int = 42, anon_seed: int = 999):
        self.game = FarmGame(rules, seed)
        self.rng = random.Random(anon_seed)
        
        # Collect all tokens
        all_tokens = set()
        for rule in rules:
            for t in rule.lhs:
                all_tokens.add(t.base)
            for o in rule.rhs:
                for t in o.tokens:
                    all_tokens.add(t.base)
        
        # Shuffle and anonymize
        tokens_list = sorted(all_tokens)
        self.rng.shuffle(tokens_list)
        self.tok2anon = {t: f"T{i:03d}" for i, t in enumerate(tokens_list)}
        self.anon2tok = {v: k for k, v in self.tok2anon.items()}
        
        # Shuffle and anonymize actions
        actions_list = [r.id for r in rules]
        self.rng.shuffle(actions_list)
        self.act2anon = {a: i for i, a in enumerate(actions_list)}
        self.anon2act = {i: a for a, i in self.act2anon.items()}
        
        self.n_actions = len(actions_list)
    
    def reset(self, seed: int = None) -> Set[str]:
        raw = self.game.reset(seed)
        return self._anon_state(raw)
    
    def _anon_state(self, raw: Dict[str, int]) -> Set[str]:
        tokens = set()
        for base, count in raw.items():
            if count > 0 and base in self.tok2anon:
                anon = self.tok2anon[base]
                # Bucket to reduce state space
                if count > 100:
                    bucket = "100+"
                elif count > 10:
                    bucket = f"{(count//10)*10}"
                else:
                    bucket = str(count)
                tokens.add(f"{anon}_{bucket}")
        return tokens
    
    def get_valid_actions(self) -> List[int]:
        raw = self.game.get_valid_actions()
        return [self.act2anon[a] for a in raw if a in self.act2anon]
    
    def step(self, action: int) -> Tuple[Set[str], float, bool, dict]:
        if action not in self.anon2act:
            return self._anon_state(self.game.state), 0, False, {}
        
        rule_id = self.anon2act[action]
        new_state, success = self.game.step(rule_id)
        
        return self._anon_state(new_state), 0, False, {'success': success}
    
    def decode_action(self, a: int) -> str:
        return self.anon2act.get(a, "???")
    
    def decode_token(self, t: str) -> str:
        base = t.split('_')[0]
        rest = t[len(base):]
        return self.anon2tok.get(base, base) + rest


if __name__ == "__main__":
    rules = parse_recipes('/mnt/user-data/uploads/recipes.csv')
    print(f"Parsed {len(rules)} rules")
    
    # Test day/night cycle
    print("\n=== Testing Day/Night Cycle ===")
    game = FarmGame(rules, seed=42)
    state = game.reset()
    
    print(f"Initial: 💧={state.get('💧',0)}, 🌞={state.get('🌞',0)}, 🌛={state.get('🌛',0)}, ☀={state.get('☀',0)}")
    
    # Check ID_Day rule
    for rule in rules:
        if rule.id == "ID_Day":
            print(f"\nID_Day LHS requirements:")
            for t in rule.lhs:
                val = state.get(t.base, 0)
                req = "any (X)" if t.consume_all else f">= {t.quantity}"
                status = "✓" if game._can_apply(rule) else "✗"
                print(f"  {t.base}: have {val}, need {req}")
            print(f"Can apply: {game._can_apply(rule)}")
    
    # Try ending the day
    if 'ID_Day' in game.get_valid_actions():
        print("\n--- Ending Day ---")
        new_state, success = game.step('ID_Day')
        print(f"Success: {success}")
        print(f"After: 💧={new_state.get('💧',0)}, 🌞={new_state.get('🌞',0)}, 🌛={new_state.get('🌛',0)}")
    
    # Test fishing with multiple seeds
    print("\n=== Testing Fishing (Probabilistic) ===")
    from collections import Counter
    fish_counter = Counter()
    
    for seed in range(50):
        game = FarmGame(rules, seed=seed)
        state = game.reset()
        
        # Fish once
        if 'ID_Rod_Fish' in game.get_valid_actions():
            new_state, _ = game.step('ID_Rod_Fish')
            
            for fish in ['🐟', '🐠', '🐡', '🦀', '🦐', '🦈']:
                if new_state.get(fish, 0) > state.get(fish, 0):
                    fish_counter[fish] += 1
    
    print(f"Fish caught across 50 seeds: {dict(fish_counter)}")
    total_fish = sum(fish_counter.values())
    if total_fish > 0:
        print("Distribution:")
        for fish, count in sorted(fish_counter.items(), key=lambda x: -x[1]):
            print(f"  {fish}: {count/total_fish*100:.1f}%")
