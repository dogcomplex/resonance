"""
SURVIVAL SIEVE: Causality from Survival Pressure (PURE VERSION)
================================================================

NO CHEATING:
- Only raw pixels (position-hashed), actions, frame numbers
- No regions (should emerge from pixel co-occurrence)
- No survival bias in coupling/decay (discover what matters)
- No magic thresholds where possible
- Survival signal is the ONLY asymmetry introduced

The sieve must discover:
- Which pixels matter
- How they relate spatially (from temporal co-occurrence)
- Which actions lead to survival in which contexts
"""

import numpy as np
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass
import hashlib


@dataclass
class SurvivalConfig:
    min_amplitude: float = 1e-12
    base_damping: float = 0.01
    coupling_rate: float = 0.01
    max_tokens: int = 1000
    context_window: int = 15


class SurvivalToken:
    """Token that tracks survival association."""

    def __init__(self, token_id: str):
        self.id = token_id
        self.amplitude: complex = complex(1e-10, 0)

        # Coupling (no survival bias - pure co-occurrence)
        self.co_occurs_with: Dict[str, float] = defaultdict(float)
        self.preceded_by: Dict[str, float] = defaultdict(float)
        self.followed_by: Dict[str, float] = defaultdict(float)

        # SPATIAL ADJACENCY: pixels know their neighbors
        # This is NOT cheating - it's the intrinsic structure of the sensory manifold
        # (you can't have pixels without a spatial arrangement)
        self.left_neighbor: Optional[str] = None
        self.right_neighbor: Optional[str] = None
        self.up_neighbor: Optional[str] = None
        self.down_neighbor: Optional[str] = None

        # Survival tracking (the ONLY place asymmetry is introduced)
        self.survive_count: int = 0
        self.die_count: int = 0

        self.activation_count: int = 0
        self._label: Optional[str] = None

    @property
    def survival_rate(self) -> float:
        """What fraction of outcomes was survival?"""
        total = self.survive_count + self.die_count
        if total == 0:
            return 0.5  # Unknown
        return self.survive_count / total

    @property
    def survival_valence(self) -> float:
        """[-1, 1] scale: -1 = always die, +1 = always survive"""
        return self.survival_rate * 2 - 1


class SurvivalSieve:
    """
    Pure survival sieve - learns from outcomes only.
    No imposed structure, no survival bias in dynamics.
    """

    def __init__(self, config: Optional[SurvivalConfig] = None):
        self.config = config or SurvivalConfig()

        self.tokens: Dict[str, SurvivalToken] = {}
        self.prev_activated: Set[str] = set()
        self.frame_count: int = 0

        # Context history: (pixel_tokens, action_token, outcome)
        self.context_history: List[Tuple[Set[str], str, Optional[bool]]] = []

        # Track action+pixel outcomes for causal learning
        self.action_pixel_outcomes: Dict[Tuple[str, str], List[bool]] = defaultdict(list)
        # Track trajectory: action+pixel+frames_before -> outcomes
        self.trajectory_outcomes: Dict[Tuple[str, str, int], List[bool]] = defaultdict(list)

        # Track MOTION patterns: action + motion_direction -> outcomes
        # motion_direction: 'left', 'right', 'up', 'down', 'static'
        self.action_motion_outcomes: Dict[Tuple[str, str], List[bool]] = defaultdict(list)

        self.damping = self.config.base_damping
        self._labels: Dict[str, str] = {}

        # Build spatial adjacency map once
        self._adjacency_built = False

    def _build_adjacency(self, h: int, w: int, stride: int):
        """Build spatial adjacency between pixel tokens."""
        if self._adjacency_built:
            return

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                pixel_hash = self._hash(f"p_{y}_{x}")
                token = self._get_token(pixel_hash)

                # Right neighbor
                if x + stride < w:
                    right_hash = self._hash(f"p_{y}_{x+stride}")
                    token.right_neighbor = right_hash
                    right_token = self._get_token(right_hash)
                    right_token.left_neighbor = pixel_hash

                # Down neighbor
                if y + stride < h:
                    down_hash = self._hash(f"p_{y+stride}_{x}")
                    token.down_neighbor = down_hash
                    down_token = self._get_token(down_hash)
                    down_token.up_neighbor = pixel_hash

        self._adjacency_built = True

    def _hash(self, data: Any) -> str:
        return hashlib.md5(str(data).encode()).hexdigest()[:16]

    def _get_token(self, token_id: str) -> SurvivalToken:
        if token_id not in self.tokens:
            if len(self.tokens) >= self.config.max_tokens:
                # Return existing token with lowest amplitude (will be reused)
                return min(self.tokens.values(), key=lambda t: abs(t.amplitude))
            self.tokens[token_id] = SurvivalToken(token_id)
        return self.tokens[token_id]

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """Observe frame + action. No regions, no magic."""
        self.frame_count = frame_num

        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float) / 255.0

        activated: Set[str] = set()
        h, w = frame.shape
        stride = 4  # Finer grid to catch motion (ball moves ~2px/frame)

        # Build spatial adjacency on first call
        self._build_adjacency(h, w, stride)

        # Pure pixel hashing - POSITION ONLY
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                intensity = frame[y, x]
                if intensity > 0.05:  # Minimal threshold - just "is something here?"
                    pixel_hash = self._hash(f"p_{y}_{x}")
                    pixel_token = self._get_token(pixel_hash)
                    pixel_token.amplitude = pixel_token.amplitude * 0.9 + complex(intensity, 0) * 0.1
                    pixel_token.activation_count += 1
                    activated.add(pixel_hash)

        # Detect MOTION: which direction did things move?
        # Motion = pixel active now that has active neighbor in prev frame
        motion_detected = {'left': 0, 'right': 0, 'up': 0, 'down': 0, 'static': 0}
        for pixel_hash in activated:
            token = self.tokens.get(pixel_hash)
            if not token:
                continue

            # Check if this pixel was also active (static)
            if pixel_hash in self.prev_activated:
                motion_detected['static'] += 1

            # Check if motion came from a neighbor
            if token.left_neighbor and token.left_neighbor in self.prev_activated:
                motion_detected['right'] += 1  # Came from left = moving right
            if token.right_neighbor and token.right_neighbor in self.prev_activated:
                motion_detected['left'] += 1   # Came from right = moving left
            if token.up_neighbor and token.up_neighbor in self.prev_activated:
                motion_detected['down'] += 1   # Came from up = moving down
            if token.down_neighbor and token.down_neighbor in self.prev_activated:
                motion_detected['up'] += 1     # Came from down = moving up

        # Store dominant motion direction for this frame
        self._current_motion = max(motion_detected.items(), key=lambda x: x[1])[0] if any(motion_detected.values()) else 'static'

        # Action token
        action_hash = self._hash(f"action_{action}")
        action_token = self._get_token(action_hash)
        action_token.amplitude = action_token.amplitude * 0.9 + complex(1.0, 0) * 0.1
        action_token.activation_count += 1
        activated.add(action_hash)

        # Learn coupling (NO survival bias - pure co-occurrence)
        self._learn_coupling(activated)

        # Store context
        self.context_history.append((activated - {action_hash}, action_hash, None))
        if len(self.context_history) > self.config.context_window * 2:
            self.context_history = self.context_history[-self.config.context_window * 2:]

        self.prev_activated = activated
        self._evolve()

    def outcome(self, survived: bool):
        """Signal outcome. This is where survival asymmetry enters."""
        # Mark recent contexts with outcome
        recent_action = None
        recent_pixels = None

        for i in range(len(self.context_history) - 1, -1, -1):
            pixels, action, out = self.context_history[i]
            if out is None:
                self.context_history[i] = (pixels, action, survived)
                if recent_action is None:
                    recent_action = action
                    recent_pixels = pixels
            else:
                break

        if not recent_action or not recent_pixels:
            return

        # Update survival counts for tokens present at outcome
        for pid in recent_pixels:
            token = self.tokens.get(pid)
            if token:
                if survived:
                    token.survive_count += 1
                else:
                    token.die_count += 1

        action_token = self.tokens.get(recent_action)
        if action_token:
            if survived:
                action_token.survive_count += 1
            else:
                action_token.die_count += 1

        # Track action+pixel outcomes
        for pid in recent_pixels:
            key = (recent_action, pid)
            self.action_pixel_outcomes[key].append(survived)
            if len(self.action_pixel_outcomes[key]) > 50:
                self.action_pixel_outcomes[key] = self.action_pixel_outcomes[key][-50:]

        # Track action+motion outcomes
        if hasattr(self, '_current_motion'):
            key = (recent_action, self._current_motion)
            self.action_motion_outcomes[key].append(survived)
            if len(self.action_motion_outcomes[key]) > 100:
                self.action_motion_outcomes[key] = self.action_motion_outcomes[key][-100:]

        # Track trajectory outcomes (action+pixel from N frames before)
        for frames_back in range(1, min(len(self.context_history), 15)):
            idx = -(frames_back + 1)
            if abs(idx) > len(self.context_history):
                break
            past_pixels, past_action, _ = self.context_history[idx]
            for pid in past_pixels:
                key = (past_action, pid, frames_back)
                self.trajectory_outcomes[key].append(survived)
                if len(self.trajectory_outcomes[key]) > 30:
                    self.trajectory_outcomes[key] = self.trajectory_outcomes[key][-30:]

    def _learn_coupling(self, activated: Set[str]):
        """Learn coupling including spatial adjacency."""
        activated_list = list(activated)

        # Co-occurrence coupling
        for i, t1 in enumerate(activated_list):
            token1 = self.tokens.get(t1)
            if not token1:
                continue

            for t2 in activated_list[i+1:]:
                token2 = self.tokens.get(t2)
                if not token2:
                    continue

                # Pure coupling - no survival modulation
                boost = self.config.coupling_rate
                token1.co_occurs_with[t2] += boost
                token2.co_occurs_with[t1] += boost

        # SPATIAL COUPLING: Adjacent pixels spread activation
        # This encodes "left-ness" and "right-ness" into the structure
        for tid in activated:
            token = self.tokens.get(tid)
            if not token:
                continue

            # Boost coupling to neighbors (directional awareness)
            for neighbor_id in [token.left_neighbor, token.right_neighbor,
                               token.up_neighbor, token.down_neighbor]:
                if neighbor_id:
                    neighbor = self.tokens.get(neighbor_id)
                    if neighbor:
                        # Spatial adjacency coupling (weaker than co-occurrence)
                        token.co_occurs_with[neighbor_id] += self.config.coupling_rate * 0.3
                        neighbor.co_occurs_with[tid] += self.config.coupling_rate * 0.3

        # TEMPORAL COUPLING - Per-pixel, not all-to-all
        # This creates the 3D space-time structure:
        # - Self-links: same pixel across frames (temporal identity)
        # - Neighbor-links: adjacent pixel was active (motion detection)
        for curr_id in activated:
            curr_token = self.tokens.get(curr_id)
            if not curr_token:
                continue

            # SELF-LINK: Same pixel position across time
            # This is the "temporal tube" - strongest link
            if curr_id in self.prev_activated:
                curr_token.preceded_by[curr_id] += self.config.coupling_rate * 2.0
                curr_token.followed_by[curr_id] += self.config.coupling_rate * 2.0

            # NEIGHBOR-LINKS: Adjacent pixel was active in previous frame
            # This detects MOTION - something moved from neighbor to here
            for neighbor_id in [curr_token.left_neighbor, curr_token.right_neighbor,
                               curr_token.up_neighbor, curr_token.down_neighbor]:
                if neighbor_id and neighbor_id in self.prev_activated:
                    neighbor_token = self.tokens.get(neighbor_id)
                    if neighbor_token:
                        # Motion from neighbor to here
                        curr_token.preceded_by[neighbor_id] += self.config.coupling_rate
                        neighbor_token.followed_by[curr_id] += self.config.coupling_rate

        # Decay
        for token in self.tokens.values():
            for k in token.co_occurs_with:
                token.co_occurs_with[k] *= 0.999
            for k in token.preceded_by:
                token.preceded_by[k] *= 0.999
            for k in token.followed_by:
                token.followed_by[k] *= 0.999

    def _evolve(self):
        """Evolve amplitudes. NO survival bias in decay."""
        new_amplitudes: Dict[str, complex] = {}

        for tid, token in self.tokens.items():
            # Pure damping - NO survival modulation
            damped = token.amplitude * (1 - self.damping)

            if abs(damped) < self.config.min_amplitude:
                damped = complex(self.config.min_amplitude, 0)

            new_amp = damped

            # Coupling contributions
            for other_id in list(token.co_occurs_with.keys())[:20]:
                other = self.tokens.get(other_id)
                if not other:
                    continue
                coupling = token.co_occurs_with[other_id]
                if coupling > 0.001:
                    transfer = coupling * other.amplitude * 0.01
                    new_amp += transfer

            new_amplitudes[tid] = new_amp

        # Normalize
        total = sum(abs(a) for a in new_amplitudes.values())
        if total > 100:
            scale = 100 / total
            new_amplitudes = {k: v * scale for k, v in new_amplitudes.items()}

        for tid, amp in new_amplitudes.items():
            if abs(amp) < self.config.min_amplitude:
                amp = complex(self.config.min_amplitude, 0)
            self.tokens[tid].amplitude = amp

    def get_action(self, action_hashes: List[str], current_context: Set[str]) -> str:
        """Choose action based on learned survival rates, especially MOTION."""
        action_scores = {}

        # Get current motion direction
        current_motion = getattr(self, '_current_motion', 'static')

        for ah in action_hashes:
            # PRIMARY: Use action+motion outcomes
            # "When ball was moving RIGHT and I took this action, did I survive?"
            motion_score = 0.0
            motion_key = (ah, current_motion)
            if motion_key in self.action_motion_outcomes:
                outcomes = self.action_motion_outcomes[motion_key]
                if len(outcomes) >= 5:
                    motion_score = sum(outcomes) / len(outcomes)

            # SECONDARY: Use trajectory knowledge
            traj_score = 0.0
            traj_count = 0
            for pid in current_context:
                for lead_time in range(1, 15):
                    key = (ah, pid, lead_time)
                    if key in self.trajectory_outcomes:
                        outcomes = self.trajectory_outcomes[key]
                        if len(outcomes) >= 3:
                            survival_rate = sum(outcomes) / len(outcomes)
                            weight = 1.0 / (1 + lead_time * 0.1)
                            traj_score += survival_rate * weight
                            traj_count += weight

            # TERTIARY: Immediate action+pixel outcomes
            imm_score = 0.0
            imm_count = 0
            for pid in current_context:
                key = (ah, pid)
                if key in self.action_pixel_outcomes:
                    outcomes = self.action_pixel_outcomes[key]
                    if len(outcomes) >= 3:
                        survival_rate = sum(outcomes) / len(outcomes)
                        imm_score += survival_rate
                        imm_count += 1

            # Combine with motion as primary signal
            if motion_score > 0:
                # Motion knowledge is most valuable
                action_scores[ah] = motion_score * 0.6
                if traj_count > 0:
                    action_scores[ah] += (traj_score / traj_count) * 0.25
                if imm_count > 0:
                    action_scores[ah] += (imm_score / imm_count) * 0.15
            elif traj_count > 0 or imm_count > 0:
                total = 0.0
                if traj_count > 0:
                    total += (traj_score / traj_count) * 0.6
                if imm_count > 0:
                    total += (imm_score / imm_count) * 0.4
                action_scores[ah] = total
            else:
                # Fall back to action's overall survival rate
                action_token = self.tokens.get(ah)
                if action_token:
                    action_scores[ah] = action_token.survival_rate
                else:
                    action_scores[ah] = 0.5

        # Choose best action, with some exploration
        if action_scores:
            best = max(action_scores.values())
            if best < 0.3:  # Uncertain - explore
                return np.random.choice(action_hashes)
            return max(action_scores.items(), key=lambda x: x[1])[0]
        return np.random.choice(action_hashes)

    def register_label(self, token_hash: str, label: str):
        self._labels[token_hash] = label
        if token_hash in self.tokens:
            self.tokens[token_hash]._label = label

    def print_state(self):
        print(f"\n{'='*70}")
        print(f"SURVIVAL SIEVE (PURE) - Frame {self.frame_count}")
        print(f"{'='*70}")
        print(f"Tokens: {len(self.tokens)}")

        # Sort by survival valence
        sorted_tokens = sorted(
            [(tid, t) for tid, t in self.tokens.items() if t.survive_count + t.die_count > 5],
            key=lambda x: x[1].survival_valence,
            reverse=True
        )

        print(f"\n--- BEST SURVIVAL (>60%) ---")
        for tid, token in sorted_tokens[:5]:
            if token.survival_rate > 0.6:
                label = self._labels.get(tid, tid[:12])
                total = token.survive_count + token.die_count
                print(f"  {label}: {token.survival_rate:.1%} ({total} outcomes)")

        print(f"\n--- WORST SURVIVAL (<40%) ---")
        for tid, token in sorted_tokens[-5:]:
            if token.survival_rate < 0.4:
                label = self._labels.get(tid, tid[:12])
                total = token.survive_count + token.die_count
                print(f"  {label}: {token.survival_rate:.1%} ({total} outcomes)")

        print(f"\n--- ACTIONS ---")
        for tid, token in self.tokens.items():
            label = self._labels.get(tid, tid[:12])
            if 'action' in label:
                total = token.survive_count + token.die_count
                if total > 0:
                    print(f"  {label}: {token.survival_rate:.1%} ({total} outcomes)")


# =============================================================================
# TEST
# =============================================================================

def test_survival_sieve():
    """Test pure survival-based learning."""
    print("=" * 70)
    print("SURVIVAL SIEVE (PURE)")
    print("No regions, no survival bias in dynamics")
    print("=" * 70)

    sieve = SurvivalSieve()

    action_hashes = {}

    ball_x, ball_y = 42.0, 20.0
    ball_dx, ball_dy = 2.0, 1.5
    paddle_x = 42.0

    hits = 0
    misses = 0

    for frame_num in range(5000):
        # Create frame
        pixels = np.zeros((84, 84), dtype=np.uint8)

        bx, by = int(ball_x), int(ball_y)
        pixels[max(0,by-2):min(84,by+2), max(0,bx-2):min(84,bx+2)] = 255

        px = int(paddle_x)
        pixels[78:82, max(0,px-8):min(84,px+8)] = 200

        pixels[0:2, :] = 100
        pixels[:, 0:2] = 100
        pixels[:, 82:84] = 100

        # Get action hashes
        for a in [0, 1, 2]:
            if a not in action_hashes:
                action_hashes[a] = sieve._hash(f"action_{a}")

        # Get current context
        current_context = sieve.prev_activated if sieve.prev_activated else set()

        # Choose action
        if frame_num < 500:
            action = np.random.randint(0, 3)
        else:
            chosen_hash = sieve.get_action(list(action_hashes.values()), current_context)
            action = [a for a, h in action_hashes.items() if h == chosen_hash][0]

        # Observe
        sieve.observe(pixels, action, frame_num)

        # Physics
        ball_x += ball_dx
        ball_y += ball_dy

        if ball_x <= 4 or ball_x >= 80:
            ball_dx *= -1
            ball_x = np.clip(ball_x, 4, 80)

        if ball_y <= 4:
            ball_dy = abs(ball_dy)

        # Outcome
        if ball_y >= 76:
            survived = abs(ball_x - paddle_x) < 12
            if survived:
                hits += 1
            else:
                misses += 1
            sieve.outcome(survived)

        if ball_y >= 78:
            ball_y = 20
            ball_x = np.random.uniform(20, 64)
            ball_dx = np.random.choice([-2, -1.5, 1.5, 2])
            ball_dy = abs(ball_dy)

        # Apply action
        if action == 0:
            paddle_x = max(10, paddle_x - 4)
        elif action == 2:
            paddle_x = min(74, paddle_x + 4)

        if (frame_num + 1) % 1000 == 0:
            for a, h in action_hashes.items():
                sieve.register_label(h, f"action_{a}")

            # Label some pixel positions for debugging
            for y in range(0, 84, 8):
                for x in range(0, 84, 8):
                    pixel_hash = sieve._hash(f"p_{y}_{x}")
                    sieve.register_label(pixel_hash, f"p_{y}_{x}")

            sieve.print_state()
            print(f"\nGame: {hits} hits, {misses} misses ({100*hits/max(1,hits+misses):.1f}%)")

            # Show action+MOTION outcomes (the key learning)
            print("\nAction + MOTION survival rates:")
            for (ah, motion), outcomes in sorted(sieve.action_motion_outcomes.items()):
                if len(outcomes) >= 10:
                    rate = sum(outcomes) / len(outcomes)
                    action_label = sieve._labels.get(ah, ah[:8])
                    print(f"  {action_label} + {motion}: {rate:.1%} ({len(outcomes)} samples)")

            # Show best action+pixel combinations
            print("\nBest action+pixel survival rates:")
            best_combos = []
            for (ah, ph), outcomes in sieve.action_pixel_outcomes.items():
                if len(outcomes) >= 10:
                    rate = sum(outcomes) / len(outcomes)
                    action_label = sieve._labels.get(ah, ah[:8])
                    pixel_label = sieve._labels.get(ph, ph[:8])
                    best_combos.append((action_label, pixel_label, rate, len(outcomes)))
            best_combos.sort(key=lambda x: x[2], reverse=True)
            for action, pixel, rate, n in best_combos[:5]:
                print(f"  {action} + {pixel}: {rate:.1%} ({n} samples)")

    return sieve


if __name__ == "__main__":
    test_survival_sieve()
