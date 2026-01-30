"""
FRACTAL PURE SIEVE: Zero Hints, Full Depth
==========================================

Building on pure_sieve.py:
1. Every token IS a sieve (fractal recursion)
2. Actions and frames are just tokens (no special treatment)
3. Goals are viewer lenses OR agent amplifiers (two modes)
4. Labels exist only for external viewing (sieve is blind to them)

Key insight: The sieve discovers structure as a GRAPH of couplings,
not as dimensions. "Dimensions" were a cheat - there are no axes,
only relationships between anonymous patterns.

VIEWER MODE vs AGENT MODE:
- Viewer: We can label tokens and highlight goal paths for OUR understanding
- Agent: Goal tokens get amplified, affecting which couplings strengthen

Both use the same underlying structure - goals are just a lens/amplifier.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import weakref


@dataclass
class FractalConfig:
    """Configuration - pure numbers, no semantics."""
    patch_size: int = 8
    n_patches: int = 10
    base_damping: float = 0.01
    min_amplitude: float = 1e-12
    coupling_strength: float = 0.1
    max_tokens: int = 500
    frame_buffer_size: int = 5
    max_depth: int = 3  # Fractal depth limit
    sub_sieve_threshold: float = 0.1  # Amplitude to spawn sub-sieve


class FractalToken:
    """
    A token that IS a sieve.

    Each token can contain a sub-sieve that captures finer structure
    about WHEN and HOW this token activates.
    """

    def __init__(self, token_id: str, depth: int = 0, max_depth: int = 3):
        self.id = token_id
        self.depth = depth
        self.max_depth = max_depth

        self.amplitude: complex = complex(1e-8, 0)

        # Co-occurrence structure (other token IDs)
        self.co_occurrence: Dict[str, float] = defaultdict(float)
        self.preceded_by: Dict[str, float] = defaultdict(float)
        self.followed_by: Dict[str, float] = defaultdict(float)

        # Activation tracking
        self.activation_count: int = 0
        self.recent_activations: List[int] = []  # Frame numbers

        # Sub-sieve (fractal recursion) - lazy initialized
        self._sub_sieve: Optional['FractalSieve'] = None

        # External label (for viewer only - sieve is blind to this)
        self._viewer_label: Optional[str] = None

    @property
    def sub_sieve(self) -> Optional['FractalSieve']:
        """Lazy sub-sieve creation when token is significant enough."""
        return self._sub_sieve

    def maybe_spawn_subsieve(self, threshold: float, config: 'FractalConfig'):
        """Spawn sub-sieve if we're significant and not at max depth."""
        if self._sub_sieve is None and self.depth < self.max_depth:
            if abs(self.amplitude) > threshold and self.activation_count > 10:
                self._sub_sieve = FractalSieve(
                    config=config,
                    depth=self.depth + 1,
                    parent_id=self.id
                )

    def set_viewer_label(self, label: str):
        """Set label for viewer (sieve doesn't see this)."""
        self._viewer_label = label

    def get_display_id(self) -> str:
        """Get ID for display (uses label if set)."""
        if self._viewer_label:
            return f"{self._viewer_label} [{self.id[:8]}]"
        return self.id[:16]


class FractalSieve:
    """
    A sieve where every token is itself a sieve.

    NO semantic labels. NO dimensions. Just patterns and couplings.
    Goals are lenses (viewer) or amplifiers (agent), not part of structure.
    """

    def __init__(self, config: Optional[FractalConfig] = None,
                 depth: int = 0, parent_id: Optional[str] = None):
        self.config = config or FractalConfig()
        self.depth = depth
        self.parent_id = parent_id

        self.tokens: Dict[str, FractalToken] = {}
        self.couplings: Dict[Tuple[str, str], float] = defaultdict(float)

        self.frame_buffer: List[np.ndarray] = []
        self.activation_buffer: List[Set[str]] = []
        self.frame_count: int = 0

        self.entropy_history: List[float] = []
        self.damping = self.config.base_damping

        # External label registry (for viewer only)
        self._label_registry: Dict[str, str] = {}  # hash -> label

        # Goal tokens (for agent mode)
        self._goal_tokens: Set[str] = set()
        self._goal_amplification: float = 1.0  # How much to boost goal-coupled tokens

    def _hash(self, data: Any) -> str:
        """Pure hash - works on any data."""
        if isinstance(data, np.ndarray):
            quantized = (data.flatten() * 10).astype(np.int8)
            return hashlib.md5(quantized.tobytes()).hexdigest()[:16]
        else:
            return hashlib.md5(str(data).encode()).hexdigest()[:16]

    def _get_token(self, token_id: str) -> FractalToken:
        """Get or create token."""
        if token_id not in self.tokens:
            if len(self.tokens) >= self.config.max_tokens:
                return self._find_similar(token_id)
            self.tokens[token_id] = FractalToken(
                token_id,
                depth=self.depth,
                max_depth=self.config.max_depth
            )
        return self.tokens[token_id]

    def _find_similar(self, new_id: str) -> FractalToken:
        """Find most activated token."""
        return max(self.tokens.values(), key=lambda t: abs(t.amplitude))

    # =========================================================================
    # VIEWER FUNCTIONS (for us to interpret - sieve is blind to these)
    # =========================================================================

    def register_label(self, token_hash: str, label: str):
        """Register a label for a token hash (viewer only)."""
        self._label_registry[token_hash] = label
        if token_hash in self.tokens:
            self.tokens[token_hash].set_viewer_label(label)

    def auto_label_from_pattern(self, pattern: np.ndarray, label: str):
        """Hash a pattern and register its label (viewer only)."""
        h = self._hash(pattern)
        self.register_label(h, label)
        return h

    def get_labeled_view(self) -> Dict[str, Any]:
        """Get structure with labels applied (for viewer)."""
        result = {
            'tokens': [],
            'couplings': [],
            'temporal_chains': [],
        }

        # Tokens with labels
        for tid, token in sorted(self.tokens.items(),
                                  key=lambda x: -abs(x[1].amplitude)):
            if abs(token.amplitude) > 0.001:
                label = self._label_registry.get(tid, tid[:12])
                result['tokens'].append({
                    'id': tid,
                    'label': label,
                    'amplitude': abs(token.amplitude),
                    'connections': len(token.co_occurrence),
                    'has_subsieve': token.sub_sieve is not None,
                })

        # Couplings with labels
        for (t1, t2), strength in sorted(self.couplings.items(),
                                          key=lambda x: -x[1])[:20]:
            if strength > 0.01:
                l1 = self._label_registry.get(t1, t1[:8])
                l2 = self._label_registry.get(t2, t2[:8])
                result['couplings'].append({
                    'from': l1,
                    'to': l2,
                    'strength': strength,
                })

        return result

    def view_from_goal(self, goal_hash: str) -> Dict[str, Any]:
        """
        View the sieve structure from a goal's perspective.

        This is VIEWER MODE: doesn't change the sieve, just shows
        what's connected to the goal.
        """
        if goal_hash not in self.tokens:
            return {'error': 'Goal token not found'}

        goal_token = self.tokens[goal_hash]

        # Find all tokens coupled to goal
        connected = []
        for tid, token in self.tokens.items():
            if tid == goal_hash:
                continue

            # Direct coupling
            coupling = self.couplings.get((tid, goal_hash), 0)
            coupling += self.couplings.get((goal_hash, tid), 0)

            # Temporal connection (what leads to goal?)
            leads_to_goal = goal_token.preceded_by.get(tid, 0)
            follows_goal = goal_token.followed_by.get(tid, 0)

            if coupling > 0.001 or leads_to_goal > 0.001 or follows_goal > 0.001:
                label = self._label_registry.get(tid, tid[:12])
                connected.append({
                    'id': tid,
                    'label': label,
                    'coupling': coupling,
                    'leads_to_goal': leads_to_goal,
                    'follows_goal': follows_goal,
                    'total_relevance': coupling + leads_to_goal * 2,  # Weight causation
                })

        connected.sort(key=lambda x: -x['total_relevance'])

        return {
            'goal': self._label_registry.get(goal_hash, goal_hash[:12]),
            'connected_tokens': connected[:30],
            'goal_amplitude': abs(goal_token.amplitude),
        }

    # =========================================================================
    # AGENT FUNCTIONS (goals affect behavior)
    # =========================================================================

    def set_goal(self, goal_hash: str, amplification: float = 2.0):
        """
        Set a goal token for agent mode.

        This DOES affect the sieve: tokens coupled to goal get amplified,
        strengthening pathways toward the goal.
        """
        self._goal_tokens.add(goal_hash)
        self._goal_amplification = amplification

    def clear_goals(self):
        """Clear all goals (return to neutral exploration)."""
        self._goal_tokens.clear()
        self._goal_amplification = 1.0

    def get_goal_directed_action(self, action_hashes: List[str]) -> Optional[str]:
        """
        Choose action based on goal coupling.

        Returns the action hash most strongly coupled to goal tokens.
        """
        if not self._goal_tokens:
            return None

        action_scores = {}
        for action_hash in action_hashes:
            if action_hash not in self.tokens:
                continue

            # Score = coupling to goals + what it leads to that's goal-coupled
            score = 0
            action_token = self.tokens[action_hash]

            for goal_hash in self._goal_tokens:
                # Direct coupling to goal
                score += self.couplings.get((action_hash, goal_hash), 0)
                score += self.couplings.get((goal_hash, action_hash), 0)

                # What does this action lead to?
                for followed_id, strength in action_token.followed_by.items():
                    # Does what it leads to couple to goal?
                    goal_coupling = self.couplings.get((followed_id, goal_hash), 0)
                    score += strength * goal_coupling

            action_scores[action_hash] = score

        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        return None

    # =========================================================================
    # CORE SIEVE FUNCTIONS (no semantics)
    # =========================================================================

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe raw data.

        Frame is just a pixel array.
        Action is just an integer.
        Frame_num is just an integer.

        ALL are treated as raw patterns to hash.
        """
        self.frame_count = frame_num

        # Normalize frame
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float) / 255.0

        # Buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.config.frame_buffer_size:
            self.frame_buffer = self.frame_buffer[-self.config.frame_buffer_size:]

        activated: Set[str] = set()

        # 1. Hash raw patches (no categorization)
        h, w = frame.shape
        patch_size = self.config.patch_size
        stride_y = h // self.config.n_patches
        stride_x = w // self.config.n_patches

        for i in range(self.config.n_patches):
            for j in range(self.config.n_patches):
                y1 = i * stride_y
                y2 = min(h, y1 + patch_size)
                x1 = j * stride_x
                x2 = min(w, x1 + patch_size)

                patch = frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue

                token_id = self._hash(patch)
                token = self._get_token(token_id)

                intensity = np.mean(patch)
                phase = intensity * np.pi
                injection = complex(intensity, 0) * np.exp(1j * phase)
                token.amplitude = token.amplitude * 0.9 + injection * 0.1
                token.activation_count += 1
                token.recent_activations.append(frame_num)

                # Maybe spawn sub-sieve
                token.maybe_spawn_subsieve(
                    self.config.sub_sieve_threshold,
                    self.config
                )

                # Feed to sub-sieve if exists
                if token.sub_sieve:
                    # Sub-sieve sees the patch as its "frame"
                    token.sub_sieve.observe_raw(patch, frame_num)

                activated.add(token_id)

        # 2. Hash frame differences
        if len(self.frame_buffer) >= 2:
            diff = np.abs(self.frame_buffer[-1] - self.frame_buffer[-2])

            for i in range(self.config.n_patches):
                for j in range(self.config.n_patches):
                    y1 = i * stride_y
                    y2 = min(h, y1 + patch_size)
                    x1 = j * stride_x
                    x2 = min(w, x1 + patch_size)

                    diff_patch = diff[y1:y2, x1:x2]
                    if diff_patch.size == 0:
                        continue

                    diff_intensity = np.mean(diff_patch)
                    if diff_intensity > 0.01:
                        token_id = self._hash(diff_patch)
                        token = self._get_token(token_id)

                        injection = complex(diff_intensity, 0)
                        token.amplitude = token.amplitude * 0.9 + injection * 0.1
                        token.activation_count += 1

                        activated.add(token_id)

        # 3. Hash second derivative
        if len(self.frame_buffer) >= 3:
            d1 = self.frame_buffer[-2] - self.frame_buffer[-3]
            d2 = self.frame_buffer[-1] - self.frame_buffer[-2]
            dd = d2 - d1

            token_id = self._hash(dd)
            token = self._get_token(token_id)

            consistency = 1.0 - np.clip(np.mean(np.abs(dd)), 0, 1)
            if consistency > 0.5:
                injection = complex(consistency, 0)
                token.amplitude = token.amplitude * 0.9 + injection * 0.1
                token.activation_count += 1
                activated.add(token_id)

        # 4. Action as anonymous token
        action_id = self._hash(np.array([action], dtype=np.int8))
        action_token = self._get_token(action_id)
        action_token.amplitude = action_token.amplitude * 0.9 + complex(0.3, 0) * 0.1
        action_token.activation_count += 1
        activated.add(action_id)

        # 5. Frame number as token (the sieve doesn't know it's "time")
        frame_id = self._hash(np.array([frame_num % 1000], dtype=np.int16))
        frame_token = self._get_token(frame_id)
        frame_token.amplitude = frame_token.amplitude * 0.9 + complex(0.1, 0) * 0.1
        frame_token.activation_count += 1
        activated.add(frame_id)

        # Learn structure
        self._learn_structure(activated)

        # Buffer activations
        self.activation_buffer.append(activated)
        if len(self.activation_buffer) > self.config.frame_buffer_size:
            self.activation_buffer = self.activation_buffer[-self.config.frame_buffer_size:]

        # Evolve
        self._evolve()

    def observe_raw(self, data: np.ndarray, frame_num: int):
        """Observe raw data without action (for sub-sieves)."""
        token_id = self._hash(data)
        token = self._get_token(token_id)

        intensity = np.mean(data) if data.size > 0 else 0
        injection = complex(intensity, 0)
        token.amplitude = token.amplitude * 0.9 + injection * 0.1
        token.activation_count += 1

    def _learn_structure(self, activated: Set[str]):
        """Learn structure from co-activation."""
        activated_list = list(activated)

        # Co-occurrence
        for i, t1 in enumerate(activated_list):
            token1 = self.tokens.get(t1)
            if not token1:
                continue

            for t2 in activated_list[i+1:]:
                token2 = self.tokens.get(t2)
                if not token2:
                    continue

                # Base coupling
                coupling_boost = 0.01

                # AGENT MODE: Boost if either token is goal-coupled
                if self._goal_tokens:
                    for goal in self._goal_tokens:
                        if (self.couplings.get((t1, goal), 0) > 0.01 or
                            self.couplings.get((t2, goal), 0) > 0.01):
                            coupling_boost *= self._goal_amplification
                            break

                self.couplings[(t1, t2)] += coupling_boost
                self.couplings[(t2, t1)] += coupling_boost

                token1.co_occurrence[t2] += coupling_boost
                token2.co_occurrence[t1] += coupling_boost

        # Temporal structure
        if len(self.activation_buffer) >= 2:
            prev_activated = self.activation_buffer[-2]

            for curr_id in activated:
                curr_token = self.tokens.get(curr_id)
                if not curr_token:
                    continue

                for prev_id in prev_activated:
                    prev_token = self.tokens.get(prev_id)
                    if not prev_token:
                        continue

                    temporal_boost = 0.005

                    # AGENT MODE: Boost temporal links toward goals
                    if self._goal_tokens and curr_id in self._goal_tokens:
                        temporal_boost *= self._goal_amplification

                    self.couplings[(prev_id, curr_id)] += temporal_boost
                    curr_token.preceded_by[prev_id] += temporal_boost
                    prev_token.followed_by[curr_id] += temporal_boost

        # Decay
        for key in self.couplings:
            self.couplings[key] *= 0.999

    def _evolve(self):
        """Evolve amplitudes."""
        new_amplitudes: Dict[str, complex] = {}

        for token_id, token in self.tokens.items():
            damped = token.amplitude * (1 - self.damping)
            if abs(damped) < self.config.min_amplitude:
                damped = complex(self.config.min_amplitude, 0)

            new_amp = damped

            # Coupling contributions
            for other_id, other in self.tokens.items():
                if other_id == token_id:
                    continue

                coupling = self.couplings.get((other_id, token_id), 0)
                if coupling > 0.0001:
                    transfer = coupling * other.amplitude * self.config.coupling_strength * 0.01
                    new_amp += transfer

            # AGENT MODE: Goal tokens get amplitude boost
            if token_id in self._goal_tokens:
                new_amp *= self._goal_amplification

            new_amplitudes[token_id] = new_amp

        # Normalize
        total = sum(abs(a) for a in new_amplitudes.values())
        if total > 100:
            scale = 100 / total
            new_amplitudes = {k: v * scale for k, v in new_amplitudes.items()}

        # Apply
        for token_id, amp in new_amplitudes.items():
            if abs(amp) < self.config.min_amplitude:
                amp = complex(self.config.min_amplitude, 0)
            self.tokens[token_id].amplitude = amp

        # Evolve sub-sieves
        for token in self.tokens.values():
            if token.sub_sieve:
                token.sub_sieve._evolve()

        self._self_tune()

    def _self_tune(self):
        """Self-tune damping."""
        amplitudes = [abs(t.amplitude) for t in self.tokens.values()
                     if abs(t.amplitude) > 0.001]

        if len(amplitudes) < 3:
            return

        total = sum(amplitudes)
        probs = [a / total for a in amplitudes]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(probs))
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        self.entropy_history.append(normalized)

        if len(self.entropy_history) < 10:
            return

        if normalized < 0.5:
            self.damping = max(0.001, self.damping * 0.95)
        elif normalized > 0.85:
            self.damping = min(0.1, self.damping * 1.05)

    def get_stats(self) -> Dict:
        """Get statistics."""
        n_with_subsieve = sum(1 for t in self.tokens.values() if t.sub_sieve)

        total_subsieve_tokens = sum(
            len(t.sub_sieve.tokens) for t in self.tokens.values()
            if t.sub_sieve
        )

        return {
            'depth': self.depth,
            'n_tokens': len(self.tokens),
            'n_significant': sum(1 for t in self.tokens.values() if abs(t.amplitude) > 0.01),
            'n_with_subsieve': n_with_subsieve,
            'total_subsieve_tokens': total_subsieve_tokens,
            'n_couplings': len([c for c in self.couplings.values() if c > 0.001]),
            'entropy': self.entropy_history[-1] if self.entropy_history else 0,
            'damping': self.damping,
            'goal_tokens': len(self._goal_tokens),
        }

    def print_state(self, show_labels: bool = True):
        """Print state."""
        stats = self.get_stats()

        print(f"\n{'='*70}")
        print(f"FRACTAL SIEVE (depth={self.depth}) - Frame {self.frame_count}")
        print(f"{'='*70}")

        print(f"\nTokens: {stats['n_tokens']} ({stats['n_significant']} significant)")
        print(f"With sub-sieves: {stats['n_with_subsieve']} (total sub-tokens: {stats['total_subsieve_tokens']})")
        print(f"Couplings: {stats['n_couplings']}")
        print(f"Entropy: {stats['entropy']:.3f}, Damping: {stats['damping']:.5f}")
        print(f"Goals set: {stats['goal_tokens']}")

        print(f"\nTop Tokens:")
        sorted_tokens = sorted(self.tokens.items(), key=lambda x: -abs(x[1].amplitude))
        for tid, token in sorted_tokens[:12]:
            if abs(token.amplitude) < 0.001:
                break

            if show_labels and tid in self._label_registry:
                display = f"{self._label_registry[tid]} [{tid[:8]}]"
            else:
                display = tid[:16]

            subsieve_info = ""
            if token.sub_sieve:
                subsieve_info = f" [subsieve: {len(token.sub_sieve.tokens)} tokens]"

            print(f"  {display}: amp={abs(token.amplitude):.4f}, "
                  f"co={len([v for v in token.co_occurrence.values() if v > 0.01])}, "
                  f"pre={len([v for v in token.preceded_by.values() if v > 0.01])}"
                  f"{subsieve_info}")


# =============================================================================
# TEST
# =============================================================================

def test_fractal_sieve():
    """Test fractal sieve with viewer labels and goal modes."""
    print("=" * 70)
    print("FRACTAL PURE SIEVE")
    print("Zero hints to sieve, labels for viewer, goals for agent")
    print("=" * 70)

    sieve = FractalSieve()

    # We'll track hashes externally for labeling
    action_hashes = {}

    # Simulation
    obj1_x, obj1_y = 42.0, 20.0
    obj1_dx, obj1_dy = 2.0, 1.5
    obj2_x = 42.0

    hits = 0
    misses = 0

    for frame_num in range(2000):
        # Create pixels
        pixels = np.zeros((84, 84), dtype=np.uint8)

        o1x, o1y = int(obj1_x), int(obj1_y)
        pixels[max(0,o1y-2):min(84,o1y+2), max(0,o1x-2):min(84,o1x+2)] = 255

        o2x = int(obj2_x)
        pixels[78:82, max(0,o2x-8):min(84,o2x+8)] = 200

        pixels[0:2, :] = 100
        pixels[:, 0:2] = 100
        pixels[:, 82:84] = 100

        # Action
        if obj1_x < obj2_x - 5:
            action = 0
        elif obj1_x > obj2_x + 5:
            action = 2
        else:
            action = 1

        if np.random.random() < 0.2:
            action = np.random.randint(0, 3)

        # Track action hash for external labeling
        action_hash = sieve._hash(np.array([action], dtype=np.int8))
        action_hashes[action] = action_hash

        # Observe
        sieve.observe(pixels, action, frame_num)

        # Physics
        obj1_x += obj1_dx
        obj1_y += obj1_dy

        if obj1_x <= 4 or obj1_x >= 80:
            obj1_dx *= -1
            obj1_x = np.clip(obj1_x, 4, 80)

        if obj1_y <= 4:
            obj1_dy = abs(obj1_dy)

        # Track outcomes for labeling
        if obj1_y >= 76:
            if abs(obj1_x - obj2_x) < 12:
                hits += 1
                # We could label the current frame's activation pattern as "hit"
            else:
                misses += 1

        if obj1_y >= 78:
            obj1_y = 20
            obj1_x = np.random.uniform(20, 64)
            obj1_dx = np.random.choice([-2, -1.5, 1.5, 2])
            obj1_dy = abs(obj1_dy)

        if action == 0:
            obj2_x = max(10, obj2_x - 4)
        elif action == 2:
            obj2_x = min(74, obj2_x + 4)

        if (frame_num + 1) % 500 == 0:
            # Register labels for viewer (sieve doesn't see these)
            for a, h in action_hashes.items():
                sieve.register_label(h, f"action_{a}")

            sieve.print_state(show_labels=True)
            print(f"\nGame: {hits} hits, {misses} misses")

    # Final analysis
    print("\n" + "=" * 70)
    print("VIEWER MODE: Structure with labels")
    print("=" * 70)

    labeled_view = sieve.get_labeled_view()
    print(f"\nLabeled tokens ({len(labeled_view['tokens'])}):")
    for t in labeled_view['tokens'][:10]:
        print(f"  {t['label']}: amp={t['amplitude']:.4f}, "
              f"connections={t['connections']}, subsieve={t['has_subsieve']}")

    # Goal-directed view
    print("\n" + "=" * 70)
    print("VIEWER MODE: View from action_1's perspective")
    print("=" * 70)

    if 1 in action_hashes:
        goal_view = sieve.view_from_goal(action_hashes[1])
        print(f"\nWhat leads to/from {goal_view['goal']}:")
        for conn in goal_view['connected_tokens'][:10]:
            print(f"  {conn['label']}: coupling={conn['coupling']:.4f}, "
                  f"leads_to_goal={conn['leads_to_goal']:.4f}")

    # Agent mode
    print("\n" + "=" * 70)
    print("AGENT MODE: Set goal and choose action")
    print("=" * 70)

    if 1 in action_hashes:
        sieve.set_goal(action_hashes[1], amplification=2.0)

        chosen = sieve.get_goal_directed_action(list(action_hashes.values()))
        for a, h in action_hashes.items():
            if h == chosen:
                print(f"Goal-directed choice: action_{a}")

    return sieve


if __name__ == "__main__":
    test_fractal_sieve()
