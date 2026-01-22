"""
V11 Proper: V9's hierarchical rules + SQLite memory + trajectory tokens + event discovery

This is V9 with three additions:
1. SQLite backend for exact memory (scalability)
2. Automatic trajectory tokens (dynamics)
3. Event discovery (game understanding)
"""

import sqlite3
import json
import hashlib
from collections import defaultdict
from typing import Set, FrozenSet, Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


def hash_state(state: FrozenSet[str]) -> str:
    return hashlib.md5(json.dumps(sorted(state)).encode()).hexdigest()


@dataclass
class Rule:
    pattern: FrozenSet[str]
    action: int
    effect: str
    successes: int = 0
    failures: int = 0
    
    @property
    def probability(self) -> float:
        n = self.successes + self.failures
        return self.successes / n if n > 0 else 0.0
    
    @property
    def observations(self) -> int:
        return self.successes + self.failures
    
    @property
    def specificity(self) -> int:
        return len(self.pattern)
    
    def record(self, occurred: bool):
        if occurred:
            self.successes += 1
        else:
            self.failures += 1


@dataclass
class QualitativeRule:
    event: str
    conditions: List[str]
    confidence: float
    support: int


class HierarchicalLearner:
    """V11: V9 + SQLite + Trajectory + Event Discovery"""
    
    def __init__(self, n_actions: int = 7,
                 db_path: str = ":memory:",
                 position_prefixes: List[str] = None,
                 prototype_prefixes: List[str] = None,
                 enable_trajectory: bool = True):
        self.n_actions = n_actions
        self.position_prefixes = position_prefixes or ['pos_']
        self.prototype_prefixes = prototype_prefixes or [
            'front_', 'left_', 'right_', 'dir_', 'carry_', 'ctx_'
        ]
        self.enable_trajectory = enable_trajectory
        
        # SQLite for exact memory
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            CREATE TABLE IF NOT EXISTS observations (
                state_hash TEXT,
                action INTEGER,
                effects_hash TEXT,
                effects_json TEXT,
                count INTEGER DEFAULT 1,
                PRIMARY KEY (state_hash, action, effects_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_sa ON observations(state_hash, action);
        """)
        self._cache: Dict[Tuple[str, int], Dict[FrozenSet[str], int]] = {}
        self._pending: List = []
        
        # V9's hierarchical rules (KEPT!)
        self.rules: Dict[Tuple[str, int], List[Rule]] = defaultdict(list)
        self.rel_rules: Dict[Tuple[str, int], List[Rule]] = defaultdict(list)
        self.effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        self.rel_effect_patterns: Dict[Tuple[str, int], List[FrozenSet]] = defaultdict(list)
        
        # V9's delta tracking
        self.delta_effects: Dict[Tuple[FrozenSet, str, int], Dict[Tuple[int,int], int]] = defaultdict(lambda: defaultdict(int))
        self.delta_counts: Dict[Tuple[FrozenSet, str, int], int] = defaultdict(int)
        
        # Trajectory tracking (NEW)
        self._prev_values: Dict[str, int] = {}
        
        # Event discovery (NEW)
        self._event_contexts: Dict[str, List[Set[str]]] = defaultdict(list)
        self._baseline_tokens: Dict[str, int] = defaultdict(int)
        self._total_obs = 0
    
    def reset_episode(self):
        """Reset trajectory tracking at episode boundary."""
        self._prev_values = {}
    
    # === Token classification (from V9) ===
    
    def _is_position_token(self, token: str) -> bool:
        return any(token.startswith(p) for p in self.position_prefixes)
    
    def _is_position_effect(self, effect: str) -> bool:
        return self._is_position_token(effect[1:])
    
    def _get_relative_state(self, state: FrozenSet) -> FrozenSet:
        return frozenset(t for t in state if not self._is_position_token(t))
    
    def _get_prototype(self, state: FrozenSet) -> FrozenSet:
        return frozenset(t for t in state if any(p in t for p in self.prototype_prefixes))
    
    def _get_direction(self, state: FrozenSet) -> Optional[str]:
        for t in state:
            if t.startswith('dir_'):
                return t
        return None
    
    def _get_position_token(self, state: FrozenSet) -> Optional[str]:
        for t in state:
            if self._is_position_token(t):
                return t
        return None
    
    def _get_position(self, state: FrozenSet) -> Optional[Tuple[int, int]]:
        for t in state:
            if t.startswith('pos_'):
                parts = t.split('_')
                if len(parts) == 3:
                    try:
                        return (int(parts[1]), int(parts[2]))
                    except ValueError:
                        pass
        return None
    
    # === Trajectory (NEW) ===
    
    def _extract_value(self, token: str) -> Tuple[Optional[str], Optional[int]]:
        match = re.match(r'(.+?)_(-?\d+)$', token)
        if match:
            return match.group(1), int(match.group(2))
        return None, None
    
    def _add_trajectory_tokens(self, state: Set[str]) -> Set[str]:
        """Add delta tokens based on changes from previous state."""
        if not self.enable_trajectory:
            return state
        
        augmented = state.copy()
        current_values = {}
        
        for token in state:
            prefix, value = self._extract_value(token)
            if prefix is not None:
                current_values[prefix] = value
                if prefix in self._prev_values:
                    delta = value - self._prev_values[prefix]
                    if delta > 0:
                        augmented.add(f"{prefix}_delta_pos")
                    elif delta < 0:
                        augmented.add(f"{prefix}_delta_neg")
        
        self._prev_values = current_values
        return augmented
    
    # === SQLite Memory ===
    
    def _flush_db(self):
        if not self._pending:
            return
        self.conn.executemany("""
            INSERT INTO observations VALUES (?, ?, ?, ?, 1)
            ON CONFLICT DO UPDATE SET count = count + 1
        """, self._pending)
        self.conn.commit()
        self._pending = []
    
    def _get_exact(self, state_fs: FrozenSet, action: int) -> Optional[Dict[FrozenSet[str], int]]:
        state_hash = hash_state(state_fs)
        cache_key = (state_hash, action)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        return None  # Only use cache, DB is for persistence
    
    def _store_exact(self, state_fs: FrozenSet, action: int, effects: FrozenSet):
        state_hash = hash_state(state_fs)
        effects_hash = hash_state(effects)
        effects_json = json.dumps(sorted(effects))
        
        # Update cache
        cache_key = (state_hash, action)
        if cache_key not in self._cache:
            self._cache[cache_key] = {}
        if effects in self._cache[cache_key]:
            self._cache[cache_key][effects] += 1
        else:
            self._cache[cache_key][effects] = 1
        
        # Queue for SQLite
        self._pending.append((state_hash, action, effects_hash, effects_json))
        if len(self._pending) >= 1000:
            self._flush_db()
    
    # === V9's Rule Engine (KEPT!) ===
    
    def _update_rules(self, rule_dict, context, action, actual_effects):
        for (effect, act), rule_list in rule_dict.items():
            if act != action:
                continue
            for rule in rule_list:
                if rule.pattern.issubset(context):
                    rule.record(effect in actual_effects)
    
    def _ensure_rules(self, effect, action, context, rule_dict, pattern_dict):
        key = (effect, action)
        patterns = pattern_dict[key]
        rules = rule_dict[key]
        existing = {r.pattern for r in rules}
        
        # Full context pattern
        patterns.append(context)
        if context not in existing:
            rules.append(Rule(pattern=context, action=action, effect=effect))
        
        # Common intersection pattern (hierarchical!)
        if len(patterns) >= 2:
            common = frozenset(set(patterns[0]).intersection(*patterns[1:]))
            if common and common not in existing:
                rules.append(Rule(pattern=common, action=action, effect=effect))
        
        # Single token patterns
        for token in context:
            single = frozenset([token])
            if single not in existing:
                rules.append(Rule(pattern=single, action=action, effect=effect))
    
    def _get_best_rule(self, rules: List[Rule], context: FrozenSet,
                        require_position: bool = False) -> Optional[Rule]:
        matching = [r for r in rules if r.pattern.issubset(context) and r.successes > 0]
        
        if require_position:
            pos_token = self._get_position_token(context)
            if pos_token:
                matching = [r for r in matching if pos_token in r.pattern]
        
        if not matching:
            return None
        
        # Most specific first (hierarchical selection!)
        matching.sort(key=lambda r: (-r.specificity, -r.observations))
        return matching[0]
    
    # === Main Interface ===
    
    def observe(self, before: Set[str], action: int, after: Set[str]) -> Dict:
        self._total_obs += 1
        
        # Raw state for exact matching (no trajectory tokens!)
        before_fs = frozenset(before)
        after_fs = frozenset(after)
        
        # Augmented state for rules (with trajectory tokens)
        before_aug = self._add_trajectory_tokens(before)
        before_aug_fs = frozenset(before_aug)
        
        # Compute effects (from raw states)
        added = after_fs - before_fs
        removed = before_fs - after_fs
        actual_effects = frozenset(
            {f"+{t}" for t in added} |
            {f"-{t}" for t in removed}
        )
        
        # Store exact observation with RAW state (not augmented!)
        self._store_exact(before_fs, action, actual_effects)
        
        # Update V9's hierarchical rules (with augmented state for trajectory info)
        pos_effects = {e for e in actual_effects if self._is_position_effect(e)}
        rel_effects = frozenset(actual_effects) - pos_effects
        before_rel = self._get_relative_state(before_aug_fs)
        
        for effect in pos_effects:
            self._ensure_rules(effect, action, before_aug_fs, self.rules, self.effect_patterns)
        for effect in rel_effects:
            self._ensure_rules(effect, action, before_rel, self.rel_rules, self.rel_effect_patterns)
        
        self._update_rules(self.rules, before_aug_fs, action, actual_effects)
        self._update_rules(self.rel_rules, before_rel, action, rel_effects)
        
        # Delta tracking (from V9) - use raw state
        direction = self._get_direction(before_fs)
        if direction:
            bp, ap = self._get_position(before_fs), self._get_position(after_fs)
            if bp and ap:
                delta = (ap[0] - bp[0], ap[1] - bp[1])
                proto = self._get_prototype(before_fs)
                key = (proto, direction, action)
                self.delta_effects[key][delta] += 1
                self.delta_counts[key] += 1
        
        # Event tracking (NEW)
        for token in before_aug:
            self._baseline_tokens[token] += 1
        self._detect_events(before_aug, after, action)
        
        return {'effects': set(actual_effects)}
    
    def _detect_events(self, before: Set[str], after: Set[str], action: int):
        """Detect and record important events."""
        for token in after:
            prefix, val = self._extract_value(token)
            if prefix and val is not None:
                # Boundary events
                if val in [0, 1, -1, -2]:
                    self._event_contexts[f"{prefix}_at_min"].append(before)
                # Could add more event types here
    
    def predict_probs(self, state: Set[str], action: int) -> Dict[str, float]:
        """Return probability predictions (V9 logic + SQLite lookup)."""
        # Raw state for exact matching
        state_fs = frozenset(state)
        
        # Augmented state for rules
        state_aug = self._add_trajectory_tokens(state.copy())
        state_aug_fs = frozenset(state_aug)
        
        # Check exact memory first (with RAW state!)
        exact = self._get_exact(state_fs, action)
        if exact:
            total = sum(exact.values())
            n_outcomes = len(exact)
            
            if n_outcomes == 1:
                effects = list(exact.keys())[0]
                return {e: 1.0 for e in effects}
            else:
                effect_counts = defaultdict(int)
                for effects, count in exact.items():
                    for e in effects:
                        effect_counts[e] += count
                return {e: c/total for e, c in effect_counts.items()}
        
        # Fall back to V9's hierarchical rules (with augmented state for trajectory)
        state_rel = self._get_relative_state(state_aug_fs)
        probs = {}
        
        # Relative rules
        for (effect, act) in self.rel_rules.keys():
            if act != action:
                continue
            best = self._get_best_rule(self.rel_rules[(effect, action)], state_rel)
            if best:
                probs[effect] = best.probability
        
        # Position rules
        for (effect, act) in self.rules.keys():
            if act != action or not self._is_position_effect(effect):
                continue
            best = self._get_best_rule(self.rules[(effect, action)], state_aug_fs, require_position=True)
            if best:
                probs[effect] = best.probability
        
        # Delta fallback (from V9) - use raw state
        if not any(self._is_position_effect(e) for e in probs):
            direction = self._get_direction(state_fs)
            current_pos = self._get_position(state_fs)
            if direction and current_pos:
                proto = self._get_prototype(state_fs)
                delta_key = (proto, direction, action)
                if delta_key in self.delta_counts and self.delta_counts[delta_key] > 0:
                    total = self.delta_counts[delta_key]
                    best_delta = max(self.delta_effects[delta_key].items(), key=lambda x: x[1])
                    delta, count = best_delta
                    prob = count / total
                    if delta != (0, 0):
                        new_pos = (current_pos[0] + delta[0], current_pos[1] + delta[1])
                        probs[f"-pos_{current_pos[0]}_{current_pos[1]}"] = prob
                        probs[f"+pos_{new_pos[0]}_{new_pos[1]}"] = prob
        
        return probs
    
    def predict(self, state: Set[str], action: int, threshold: float = 0.5) -> Set[str]:
        probs = self.predict_probs(state, action)
        return {e for e, p in probs.items() if p >= threshold}
    
    def discover_rules(self, min_support: int = 20) -> List[QualitativeRule]:
        """Discover qualitative rules from events."""
        rules = []
        for event, contexts in self._event_contexts.items():
            if len(contexts) < min_support:
                continue
            
            token_counts = defaultdict(int)
            for ctx in contexts:
                for t in ctx:
                    token_counts[t] += 1
            
            predictive = []
            for token, count in token_counts.items():
                p_given_event = count / len(contexts)
                p_baseline = self._baseline_tokens.get(token, 1) / max(1, self._total_obs)
                lift = p_given_event / max(0.001, p_baseline)
                if lift > 2 and count >= min_support // 2:
                    predictive.append((token, p_given_event, lift))
            
            if predictive:
                predictive.sort(key=lambda x: -x[2])
                rules.append(QualitativeRule(
                    event=event,
                    conditions=[t[0] for t in predictive[:5]],
                    confidence=sum(t[1] for t in predictive[:5]) / min(5, len(predictive)),
                    support=len(contexts)
                ))
        
        return rules
    
    def explain_game(self) -> str:
        rules = self.discover_rules()
        if not rules:
            return "Not enough data yet."
        lines = ["Discovered Rules:", "=" * 40]
        for r in sorted(rules, key=lambda x: -x.support):
            lines.append(f"\n{r.event}: {', '.join(r.conditions[:3])}")
            lines.append(f"  Confidence: {r.confidence:.0%}, Support: {r.support}")
        return "\n".join(lines)
    
    def stats(self) -> Dict:
        return {
            'observations': self._total_obs,
            'cached_states': len(self._cache),
            'rules': sum(len(rl) for rl in self.rules.values()),
            'rel_rules': sum(len(rl) for rl in self.rel_rules.values()),
        }
    
    def close(self):
        self._flush_db()
        self.conn.close()


if __name__ == "__main__":
    learner = HierarchicalLearner()
    learner.observe({'a', 'b'}, 0, {'a', 'c'})
    print("V11 Proper OK")
    print(learner.stats())
    learner.close()
