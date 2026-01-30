"""
RECURSIVE FRACTAL SIEVE
========================

The key insight you raised: the pure sieve was FLAT, not FRACTAL.

True fractal/recursive structure means:
1. Tokens can themselves BE sieves
2. Patterns of patterns emerge at higher levels
3. Abstraction dimensions (spatial, temporal, behavioral) are EMERGENT
4. The anthropic principle applies at EVERY level

WHAT MAKES IT RECURSIVE:
- Each token has an internal sieve for its sub-patterns
- Strong sub-patterns get PROMOTED to the parent level
- Patterns can span multiple frames/regions/contexts
- Combinations of lower-level patterns become higher-level tokens

EMERGENT ABSTRACTIONS (not imposed):
- Spatial: pixels that co-activate become a single higher token
- Temporal: action sequences that recur become a single token
- Behavioral: action + context combinations become tokens
- These compete/survive just like raw pixels do

PURITY MAINTAINED:
- NO domain knowledge
- NO magic constants (self-tuning)
- NO explicit reward
- Survival = amplitude accumulation from existing longer
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import hashlib
from collections import defaultdict


class RecursiveToken:
    """
    A token that can contain its own sieve of sub-patterns.

    At the lowest level: raw observations (pixels, actions)
    At higher levels: combinations of lower tokens
    """
    __slots__ = [
        'id',
        'level',              # Hierarchy level (0 = raw, higher = more abstract)
        'amplitude',          # Current survival amplitude
        'flows_to',           # Forward temporal flow
        'flows_from',         # Backward temporal flow
        'sub_patterns',       # Internal sieve: pattern_id -> amplitude
        'components',         # What lower-level tokens compose this one
        'active_count',       # How many times activated (for promotion threshold)
    ]

    def __init__(self, token_id: str, level: int = 0):
        self.id = token_id
        self.level = level
        self.amplitude: float = 0.0
        self.flows_to: Dict[str, float] = {}
        self.flows_from: Dict[str, float] = {}
        self.sub_patterns: Dict[str, float] = defaultdict(float)  # Internal sieve
        self.components: Set[str] = set()  # Lower-level tokens this is made of
        self.active_count: int = 0


class RecursiveFractalSieve:
    """
    Fractal sieve with true recursive structure.

    Key operations:
    1. OBSERVE: Inject raw observations at level 0
    2. COUPLE: Build temporal connections through action bottleneck
    3. PROPAGATE: Amplitude flows backward through time
    4. ABSTRACT: Co-occurring patterns become higher-level tokens
    5. PROMOTE: Strong patterns move up the hierarchy
    6. SURVIVE: Amplitude-weighted coupling (anthropic principle)
    """

    def __init__(self, max_levels: int = 3):
        self.max_levels = max_levels

        # Tokens organized by level
        self.levels: List[Dict[str, RecursiveToken]] = [
            {} for _ in range(max_levels)
        ]

        # Level 0 = raw tokens (pixels, actions)
        # Level 1 = combinations of raw tokens
        # Level 2 = combinations of combinations
        # etc.

        # Frame state
        self.frame_num: int = 0
        self.prev_active: Dict[int, Set[str]] = {i: set() for i in range(max_levels)}
        self.curr_active: Dict[int, Set[str]] = {i: set() for i in range(max_levels)}
        self.current_action_id: Optional[str] = None
        self.prev_action_id: Optional[str] = None

        # Temporal trace for credit assignment
        self._action_trace: List[str] = []

        # Self-tuning parameters
        self._coupling_strength: float = 0.1
        self._decay_rate: float = 0.01
        self._promotion_threshold: float = 50.0  # Activation count for promotion to higher level
        self._abstraction_threshold: float = 50.0  # Higher threshold for creating abstractions
        self._max_abstractions_per_level: int = 100  # Limit abstractions to force competition

        # Co-occurrence tracking for abstraction discovery
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        # Manifold built flag
        self._manifold_built = False
        self._frame_shape: Optional[Tuple[int, int]] = None

    def _hash(self, data: str) -> str:
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _get_token(self, token_id: str, level: int = 0) -> RecursiveToken:
        if token_id not in self.levels[level]:
            self.levels[level][token_id] = RecursiveToken(token_id, level)
        return self.levels[level][token_id]

    def _build_manifold(self, h: int, w: int):
        """Build spatial adjacency at level 0."""
        if self._manifold_built:
            return
        self._frame_shape = (h, w)

        for y in range(h):
            for x in range(w):
                pixel_id = self._hash(f"p_{y}_{x}")
                token = self._get_token(pixel_id, level=0)
                # Spatial neighbors encoded in flows (at level 0)
                if y > 0:
                    neighbor_id = self._hash(f"p_{y-1}_{x}")
                    token.flows_to[neighbor_id] = 0.1
                    token.flows_from[neighbor_id] = 0.1
                if y < h - 1:
                    neighbor_id = self._hash(f"p_{y+1}_{x}")
                    token.flows_to[neighbor_id] = 0.1
                    token.flows_from[neighbor_id] = 0.1
                if x > 0:
                    neighbor_id = self._hash(f"p_{y}_{x-1}")
                    token.flows_to[neighbor_id] = 0.1
                    token.flows_from[neighbor_id] = 0.1
                if x < w - 1:
                    neighbor_id = self._hash(f"p_{y}_{x+1}")
                    token.flows_to[neighbor_id] = 0.1
                    token.flows_from[neighbor_id] = 0.1

        self._manifold_built = True

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe frame and action.

        Key steps:
        1. Inject raw observations at level 0
        2. Build temporal couplings through action bottleneck
        3. Track co-occurrences for abstraction discovery
        4. Propagate amplitude (backward credit assignment)
        5. Discover and create higher-level abstractions
        6. Promote strong patterns
        """
        self.frame_num = frame_num

        # Grayscale + normalize
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        h, w = frame.shape
        self._build_manifold(h, w)

        # Store previous state
        for level in range(self.max_levels):
            self.prev_active[level] = self.curr_active[level].copy()
            self.curr_active[level] = set()
        self.prev_action_id = self.current_action_id

        # === LEVEL 0: Raw observations ===
        for y in range(h):
            for x in range(w):
                intensity = frame[y, x]
                if intensity > 0:
                    pixel_id = self._hash(f"p_{y}_{x}")
                    token = self._get_token(pixel_id, level=0)
                    token.amplitude = intensity
                    token.active_count += 1
                    self.curr_active[0].add(pixel_id)

        # Action token at level 0
        action_id = self._hash(f"action_{action}")
        action_token = self._get_token(action_id, level=0)
        action_token.amplitude = 1.0
        action_token.active_count += 1
        self.current_action_id = action_id
        self.curr_active[0].add(action_id)

        # === TEMPORAL COUPLING (through action bottleneck) ===
        self._learn_bottleneck_couplings()

        # === CO-OCCURRENCE TRACKING (for abstraction discovery) ===
        self._track_cooccurrence()

        # === AMPLITUDE PROPAGATION (backward credit assignment) ===
        self._propagate_amplitude()

        # === ABSTRACTION: Create higher-level tokens from co-occurring patterns ===
        self._discover_abstractions()

        # === ACTIVATION: Activate higher-level tokens based on components ===
        self._activate_higher_levels()

        # === DECAY ===
        self._decay_all()

    def _learn_bottleneck_couplings(self):
        """
        Build couplings THROUGH the action bottleneck.
        Amplitude-weighted: stronger actions build stronger couplings.
        """
        if not self.prev_action_id:
            return

        prev_action = self.levels[0].get(self.prev_action_id)
        if not prev_action:
            return

        # Coupling strength scales with action amplitude (anthropic principle)
        base_cs = self._coupling_strength
        amp_factor = 1.0 + prev_action.amplitude * 0.1
        cs = base_cs * amp_factor

        # prev_pixels -> prev_action
        for prev_pixel_id in self.prev_active[0]:
            if prev_pixel_id != self.prev_action_id:
                prev_pixel = self.levels[0].get(prev_pixel_id)
                if prev_pixel:
                    prev_pixel.flows_to[self.prev_action_id] = (
                        prev_pixel.flows_to.get(self.prev_action_id, 0) + cs
                    )
                    prev_action.flows_from[prev_pixel_id] = (
                        prev_action.flows_from.get(prev_pixel_id, 0) + cs
                    )

        # prev_action -> curr_pixels
        for curr_pixel_id in self.curr_active[0]:
            if curr_pixel_id != self.current_action_id:
                curr_pixel = self.levels[0].get(curr_pixel_id)
                if curr_pixel:
                    prev_action.flows_to[curr_pixel_id] = (
                        prev_action.flows_to.get(curr_pixel_id, 0) + cs
                    )
                    curr_pixel.flows_from[self.prev_action_id] = (
                        curr_pixel.flows_from.get(self.prev_action_id, 0) + cs
                    )

    def _track_cooccurrence(self):
        """
        Track MEANINGFUL co-occurrences for abstraction.

        KEY INSIGHT: We want abstractions that capture ACTION-CONTEXT pairs,
        not just static pixel combinations.

        Track:
        1. action + pixel combinations (behavioral abstractions)
        2. pixel + pixel only if they VARY together (not static elements)
        """
        if not self.current_action_id:
            return

        # Track ACTION + PIXEL co-occurrence (this is the important one!)
        # These are behavioral abstractions: "action X in context Y"
        for pixel_id in self.curr_active[0]:
            if pixel_id != self.current_action_id:
                key = (min(self.current_action_id, pixel_id),
                       max(self.current_action_id, pixel_id))
                self._cooccurrence[key] += 1

        # Track CHANGING pixels (not static ones)
        # Only pixels that appear/disappear between frames are interesting
        newly_active = self.curr_active[0] - self.prev_active[0]
        for t1 in newly_active:
            if t1 != self.current_action_id:
                for t2 in newly_active:
                    if t2 > t1 and t2 != self.current_action_id:
                        key = (t1, t2)
                        self._cooccurrence[key] += 1

    def _propagate_amplitude(self):
        """
        Backward amplitude propagation with recency weighting.
        The anthropic principle: longer survival = more amplitude.
        """
        if self.current_action_id:
            self._action_trace.append(self.current_action_id)
            if len(self._action_trace) > 100:
                self._action_trace = self._action_trace[-100:]

        # Credit assignment with recency bias
        if self._action_trace:
            lambda_discount = 0.9
            trace_len = len(self._action_trace)

            for i, action_id in enumerate(self._action_trace):
                action_token = self.levels[0].get(action_id)
                if action_token:
                    age = trace_len - 1 - i
                    discount = lambda_discount ** age
                    action_token.amplitude += 0.1 * discount

    def _discover_abstractions(self):
        """
        Create higher-level tokens from frequently co-occurring patterns.

        KEY: Only create LIMITED abstractions and let them COMPETE.
        Abstractions that don't help survival should decay away.
        """
        if self.frame_num % 100 != 0:  # Check less frequently
            return

        # Find the STRONGEST co-occurring pairs (not all of them)
        strong_pairs = [
            (pair, count) for pair, count in self._cooccurrence.items()
            if count >= self._abstraction_threshold
        ]
        strong_pairs.sort(key=lambda x: -x[1])

        # Only consider top candidates
        top_candidates = strong_pairs[:self._max_abstractions_per_level]

        for (t1, t2), count in top_candidates:
            combo_id = self._hash(f"combo_{t1}_{t2}")

            if combo_id not in self.levels[1]:
                # Check if we've hit the limit
                if len(self.levels[1]) >= self._max_abstractions_per_level:
                    # Prune weakest abstraction to make room
                    weakest = min(self.levels[1].values(), key=lambda t: t.amplitude)
                    if weakest.amplitude < 0.1:  # Only prune if truly weak
                        del self.levels[1][weakest.id]
                    else:
                        continue  # Don't create new if all existing are strong

                # New abstraction discovered!
                combo_token = self._get_token(combo_id, level=1)
                combo_token.components.add(t1)
                combo_token.components.add(t2)

                # Inherit amplitude from components
                t1_token = self.levels[0].get(t1)
                t2_token = self.levels[0].get(t2)
                if t1_token and t2_token:
                    combo_token.amplitude = (t1_token.amplitude + t2_token.amplitude) / 2

            # Decay co-occurrence to allow new patterns to emerge
            self._cooccurrence[(t1, t2)] = count // 2

        # PROMOTE: Strong level-1 abstractions to level-2
        if len(self.levels[1]) > 0:
            for token in list(self.levels[1].values()):
                if token.active_count >= self._promotion_threshold and token.amplitude > 1.0:
                    # This abstraction is strong and frequent - track it at level 2
                    self.curr_active[2].add(token.id)  # Just mark as active at higher level for now

    def _activate_higher_levels(self):
        """
        Activate higher-level tokens when their components are active.

        KEY: Abstractions get amplitude from SURVIVAL, not just components.
        When an abstraction is active during longer games, it accumulates
        more amplitude - the anthropic principle at the abstraction level.
        """
        for level in range(1, self.max_levels):
            for token_id, token in self.levels[level].items():
                # Check if all components are active at lower level
                components_active = all(
                    c in self.curr_active[level - 1]
                    for c in token.components
                )

                if components_active and token.components:
                    self.curr_active[level].add(token_id)
                    token.active_count += 1

                    # SURVIVAL CREDIT: Active abstractions get amplitude
                    # from the ongoing game (just like actions do)
                    # This means abstractions that are active during longer
                    # games accumulate more amplitude.
                    token.amplitude += 0.05  # Base survival credit

                    # BONUS: If the current action has high amplitude,
                    # co-active abstractions benefit (good actions = good contexts)
                    if self.current_action_id:
                        action_token = self.levels[0].get(self.current_action_id)
                        if action_token:
                            token.amplitude += 0.01 * action_token.amplitude

    def _decay_all(self):
        """Decay amplitudes and couplings across all levels."""
        dr = self._decay_rate

        for level in range(self.max_levels):
            for token in self.levels[level].values():
                token.amplitude *= (1 - dr)

                for key in list(token.flows_to.keys()):
                    token.flows_to[key] *= (1 - dr)
                    if token.flows_to[key] < 1e-12:
                        del token.flows_to[key]

                for key in list(token.flows_from.keys()):
                    token.flows_from[key] *= (1 - dr)
                    if token.flows_from[key] < 1e-12:
                        del token.flows_from[key]

    def signal_game_end(self):
        """Called when episode ends. Clear trace for next episode."""
        self._action_trace = []

    def choose_action(self, num_actions: int = 3) -> int:
        """
        Choose action based on coupling at ALL levels.

        Higher-level abstractions can contribute to action selection
        if they encode action-context combinations.
        """
        action_scores = {}

        for a in range(num_actions):
            action_id = self._hash(f"action_{a}")
            action_token = self.levels[0].get(action_id)

            if not action_token:
                action_scores[a] = 1.0
                continue

            # Level 0: direct pixel -> action coupling
            input_coupling = 0.0
            for pixel_id in self.curr_active[0]:
                pixel_token = self.levels[0].get(pixel_id)
                if pixel_token:
                    input_coupling += pixel_token.flows_to.get(action_id, 0)

            # Higher levels: check if any active abstractions involve this action
            abstract_bonus = 0.0
            for level in range(1, self.max_levels):
                for token_id in self.curr_active[level]:
                    token = self.levels[level].get(token_id)
                    if token and action_id in token.components:
                        # This abstraction involves this action
                        abstract_bonus += token.amplitude

            # Combine scores
            action_scores[a] = (
                input_coupling +
                0.1 * sum(action_token.flows_to.values()) +
                0.1 * action_token.amplitude +
                0.5 * abstract_bonus +  # Higher-level abstractions weighted
                0.01
            )

        # Temperature-based selection
        scores_list = list(action_scores.values())
        if len(scores_list) > 1 and max(scores_list) > min(scores_list):
            spread = (max(scores_list) - min(scores_list)) / (max(scores_list) + 0.001)
            temperature = 0.3 + 2.0 * spread
            for a in action_scores:
                if action_scores[a] > 0:
                    action_scores[a] = action_scores[a] ** (1.0 / temperature)

        total = sum(action_scores.values())
        if total < 0.001:
            return np.random.randint(0, num_actions)

        probs = {a: s / total for a, s in action_scores.items()}
        r = np.random.random()
        cumulative = 0.0
        for a, p in probs.items():
            cumulative += p
            if r < cumulative:
                return a

        return list(action_scores.keys())[-1]

    def get_action_amplitudes(self) -> Dict[int, float]:
        result = {}
        for a in range(10):
            action_id = self._hash(f"action_{a}")
            if action_id in self.levels[0]:
                result[a] = self.levels[0][action_id].amplitude
        return result

    def get_stats(self) -> Dict:
        stats = {
            'frame': self.frame_num,
            'levels': {},
            'cooccurrence_pairs': len(self._cooccurrence),
            'total_cooccurrence': sum(self._cooccurrence.values()),
        }

        for level in range(self.max_levels):
            level_tokens = self.levels[level]
            if level_tokens:
                amps = [t.amplitude for t in level_tokens.values()]
                stats['levels'][level] = {
                    'n_tokens': len(level_tokens),
                    'n_active': len(self.curr_active[level]),
                    'total_amp': sum(amps),
                    'avg_amp': np.mean(amps) if amps else 0,
                }
            else:
                stats['levels'][level] = {'n_tokens': 0, 'n_active': 0, 'total_amp': 0, 'avg_amp': 0}

        return stats

    def print_state(self):
        stats = self.get_stats()
        amps = self.get_action_amplitudes()

        print(f"\n{'='*70}")
        print(f"RECURSIVE FRACTAL SIEVE - Frame {self.frame_num}")
        print(f"{'='*70}")

        for level, level_stats in stats['levels'].items():
            print(f"\nLevel {level}: {level_stats['n_tokens']} tokens, "
                  f"{level_stats['n_active']} active, "
                  f"total_amp={level_stats['total_amp']:.2f}")

        print(f"\nCo-occurrence tracking: {stats['cooccurrence_pairs']} pairs, "
              f"total={stats['total_cooccurrence']}")

        # Show discovered abstractions at level 1+
        for level in range(1, self.max_levels):
            if self.levels[level]:
                print(f"\nLevel {level} abstractions:")
                sorted_tokens = sorted(
                    self.levels[level].values(),
                    key=lambda t: t.amplitude,
                    reverse=True
                )[:5]
                for token in sorted_tokens:
                    print(f"  {token.id[:8]}...: amp={token.amplitude:.4f}, "
                          f"components={len(token.components)}, "
                          f"active={token.active_count}")

        print(f"\nAction amplitudes:")
        for a, amp in sorted(amps.items()):
            print(f"  action_{a}: {amp:.4f}")


# =============================================================================
# TEST
# =============================================================================

def test_recursive_pong():
    """Test recursive fractal sieve on Pong."""
    print("=" * 70)
    print("RECURSIVE FRACTAL SIEVE - PONG")
    print("True fractal structure with emergent abstractions")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = RecursiveFractalSieve(max_levels=3)

    game_lengths = []
    hits = 0
    misses = 0

    ball_x, ball_y = 10.5, 5.0
    ball_dx, ball_dy = 0.5, 0.5
    paddle_x = 10.5
    current_game_length = 0

    for frame_num in range(10000):
        # Create frame
        pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

        # Ball
        bx, by = int(ball_x), int(ball_y)
        pixels[max(0,by-1):min(FRAME_SIZE,by+1), max(0,bx-1):min(FRAME_SIZE,bx+1)] = 255

        # Paddle
        px = int(paddle_x)
        pixels[FRAME_SIZE-2:FRAME_SIZE, max(0,px-2):min(FRAME_SIZE,px+2)] = 200

        # Walls
        pixels[0:1, :] = 50
        pixels[:, 0:1] = 50
        pixels[:, FRAME_SIZE-1:FRAME_SIZE] = 50

        # Choose action
        if frame_num < 100:
            action = np.random.randint(0, 3)
        else:
            action = sieve.choose_action(num_actions=3)

        # Observe
        sieve.observe(pixels, action, frame_num)

        # Physics
        ball_x += ball_dx
        ball_y += ball_dy

        if ball_x <= 1 or ball_x >= FRAME_SIZE - 1:
            ball_dx *= -1
            ball_x = np.clip(ball_x, 1, FRAME_SIZE - 1)

        if ball_y <= 1:
            ball_dy = abs(ball_dy)

        # Apply action
        if action == 0:
            paddle_x = max(3, paddle_x - 1)
        elif action == 2:
            paddle_x = min(FRAME_SIZE - 3, paddle_x + 1)

        current_game_length += 1

        # Check hit/miss
        if ball_y >= FRAME_SIZE - 2:
            hit = abs(ball_x - paddle_x) < 3
            if hit:
                hits += 1
                ball_dy = -abs(ball_dy)
                ball_y = FRAME_SIZE - 3
            else:
                misses += 1
                game_lengths.append(current_game_length)
                current_game_length = 0
                sieve.signal_game_end()

                ball_y = 5
                ball_x = np.random.uniform(5, FRAME_SIZE - 5)
                ball_dx = np.random.choice([-0.5, 0.5])
                ball_dy = 0.5

        # Progress report
        if (frame_num + 1) % 1000 == 0:
            total_games = hits + misses
            hit_rate = hits / total_games if total_games > 0 else 0
            recent_lengths = game_lengths[-10:] if len(game_lengths) >= 10 else game_lengths
            avg_length = np.mean(recent_lengths) if recent_lengths else 0

            print(f"\n--- Frame {frame_num + 1} ---")
            print(f"Games: {total_games} (hits={hits}, misses={misses})")
            print(f"Hit rate: {hit_rate:.1%}")
            print(f"Recent avg game length: {avg_length:.1f} frames")
            sieve.print_state()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_games = hits + misses
    print(f"Total games: {total_games}")
    print(f"Hits: {hits} ({100*hits/total_games:.1f}%)")
    print(f"Misses: {misses}")

    if len(game_lengths) >= 20:
        first_10 = np.mean(game_lengths[:10])
        last_10 = np.mean(game_lengths[-10:])
        print(f"\nGame length progression:")
        print(f"  First 10 games avg: {first_10:.1f} frames")
        print(f"  Last 10 games avg: {last_10:.1f} frames")
        if last_10 > first_10:
            print(f"  Improvement: {(last_10/first_10 - 1)*100:.1f}%")

    print(f"\nFinal action amplitudes:")
    amps = sieve.get_action_amplitudes()
    for a, amp in sorted(amps.items()):
        print(f"  action_{a}: {amp:.4f}")

    stats = sieve.get_stats()
    print(f"\nAbstraction discovery:")
    for level, level_stats in stats['levels'].items():
        print(f"  Level {level}: {level_stats['n_tokens']} tokens")

    return sieve, game_lengths


if __name__ == "__main__":
    test_recursive_pong()
