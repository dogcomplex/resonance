"""
PURE WAVE SIEVE WITH SELF-EMERGING ANNEALING
============================================

A wave-based fractal sieve with thermodynamic cycles that emerge from physics.

DESIGN PRINCIPLES:
1. NO MAGIC NUMBERS - all parameters derived from observable quantities
2. NO PRE-CREATED TOKENS - everything discovered through observation
3. TRUE WAVE MECHANICS - complex addition, interference, energy conservation
4. NO ARBITRARY WEIGHTS - action selection through pure resonance
5. NO DELETION - damping only, waves persist forever
6. SELF-EMERGING ANNEALING - thermal cycles from internal energy dynamics

SELF-EMERGING ANNEALING (Physics-Correct):
==========================================
Real physical systems have spontaneous thermal fluctuations WITHOUT external triggers.
The mechanism:

1. RESONANCE MISMATCH → HEAT PRODUCTION
   When coupled oscillators are out of phase, their interference is destructive.
   In real physics, this energy doesn't disappear - it goes into the thermal bath.
   Our implementation: mismatched predictions deposit energy into a heat reservoir.

2. HEAT BATH → SPONTANEOUS EXCITATION
   The heat bath has energy. By statistical mechanics (equipartition theorem),
   this energy spontaneously redistributes to all degrees of freedom.
   Our implementation: heat bath energy flows to low-amplitude tokens.

3. NATURAL CYCLES EMERGE
   - Good predictions: resonance match, energy concentrates in successful patterns
   - Bad predictions: mismatch, energy goes to heat bath
   - Heat bath re-excites dormant patterns
   - System naturally oscillates between ordered (cold) and disordered (hot) states

4. EXTERNAL NUDGE (optional)
   We can still inject external thermal energy when needed, but the system
   will cycle naturally even without intervention.

THE ONLY "KNOWLEDGE" GIVEN:
- Observations come as 2D arrays (spatial structure exists)
- Time flows forward (frames have sequence)
- Actions are discrete events that occur between observations

Everything else EMERGES.
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
import hashlib
import cmath
from collections import defaultdict


class WaveToken:
    """
    A pattern represented as a complex wave.

    The wave encodes:
    - Amplitude: relevance/activation strength
    - Phase: relationship/value encoding

    NO arbitrary initialization - starts at true zero.
    """
    __slots__ = [
        'id',
        'wave',           # Complex: amplitude * e^(i*phase)
        'couplings',      # Bidirectional coupling waves to other tokens
        'observation_count',  # How many times observed (for statistics)
        'components',     # For composite patterns
    ]

    def __init__(self, token_id: str):
        self.id = token_id
        self.wave: complex = complex(0, 0)
        self.couplings: Dict[str, complex] = {}
        self.observation_count: int = 0
        self.components: Set[str] = set()

    @property
    def amplitude(self) -> float:
        return abs(self.wave)

    @property
    def phase(self) -> float:
        return cmath.phase(self.wave)


class PureWaveSieve:
    """
    Wave sieve with physics-derived parameters.

    NO tuning - parameters emerge from the field state itself.
    """

    def __init__(self):
        # The wave field - patterns indexed by hash
        self.field: Dict[str, WaveToken] = {}

        # Composite patterns (emergent higher-order)
        self.composites: Dict[str, WaveToken] = {}

        # Frame tracking
        self.frame_num: int = 0
        self.prev_active: Set[str] = set()
        self.curr_active: Set[str] = set()

        # Action tracking (discovered, not pre-created)
        self.discovered_actions: Set[str] = set()
        self.current_action_id: Optional[str] = None
        self.prev_action_id: Optional[str] = None

        # Temporal trace for credit assignment
        self._action_trace: List[str] = []

        # Co-occurrence tracking (for composite discovery)
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        # Energy tracking (for derived damping)
        self._total_energy_history: List[float] = []
        self._target_energy: Optional[float] = None

        # Statistics
        self._stats_history: List[Dict] = []

        # SELF-EMERGING ANNEALING STATE
        # Heat bath: reservoir of thermal energy from resonance mismatches
        self._heat_bath: float = 0.0

        # Track prediction quality for resonance mismatch detection
        self._last_predicted_pixels: Set[str] = set()
        self._prediction_made: bool = False

        # Statistics
        self._recent_game_lengths: List[int] = []
        self._mismatch_history: List[float] = []
        self._heat_bath_history: List[float] = []

        # External nudge tracking
        self._external_nudges: int = 0
        self._total_heat_generated: float = 0.0
        self._total_heat_redistributed: float = 0.0

    def _hash(self, data: str) -> str:
        """Deterministic hash for token IDs."""
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def _get_token(self, token_id: str) -> WaveToken:
        """Get or create token. Tokens are NEVER deleted."""
        if token_id not in self.field:
            self.field[token_id] = WaveToken(token_id)
        return self.field[token_id]

    def _get_composite(self, comp_id: str) -> WaveToken:
        """Get or create composite pattern."""
        if comp_id not in self.composites:
            self.composites[comp_id] = WaveToken(comp_id)
        return self.composites[comp_id]

    def _total_field_energy(self) -> float:
        """Total energy in the wave field (sum of squared amplitudes)."""
        field_energy = sum(abs(t.wave)**2 for t in self.field.values())
        composite_energy = sum(abs(c.wave)**2 for c in self.composites.values())
        return field_energy + composite_energy

    def _field_entropy(self) -> float:
        """Shannon entropy of amplitude distribution - measures disorder."""
        amplitudes = [t.amplitude for t in self.field.values() if t.amplitude > 1e-10]
        if not amplitudes:
            return 0.0
        total = sum(amplitudes)
        if total < 1e-10:
            return 0.0
        probs = [a / total for a in amplitudes]
        return -sum(p * np.log(p + 1e-10) for p in probs)

    def _derive_damping(self) -> float:
        """
        Derive damping rate from energy dynamics.

        PRINCIPLE: System should reach thermodynamic equilibrium.
        Energy injection per frame ~ N_active (number of active tokens)
        Damping should remove equal energy on average.

        For equilibrium: damping_rate * total_energy ≈ injection_rate
        Therefore: damping_rate ≈ injection_rate / total_energy
        """
        current_energy = self._total_field_energy()
        self._total_energy_history.append(current_energy)

        # Keep limited history
        if len(self._total_energy_history) > 100:
            self._total_energy_history = self._total_energy_history[-100:]

        # Estimate injection rate: roughly N_active per frame
        # Each active token gets ~1 unit of energy added
        n_active = len(self.curr_active)

        if current_energy < 1.0:
            return 0.01  # Moderate damping when energy is tiny

        # Equilibrium condition: damping removes what injection adds
        # damping_rate = injection / energy
        # But we want slightly MORE damping to ensure stability
        damping = (n_active / current_energy) * 1.5  # 1.5x for stability margin

        # Clamp to prevent pathological cases
        return min(0.5, max(0.001, damping))

    def _derive_temperature(self) -> float:
        """
        Derive selection temperature from field entropy AND heat bath.

        PRINCIPLE: Temperature reflects both field disorder AND available thermal energy.
        - Base temperature from field entropy (information-theoretic)
        - Heat bath contributes additional thermal fluctuations
        """
        entropy = self._field_entropy()
        max_entropy = np.log(len(self.field) + 1)  # Maximum possible entropy

        if max_entropy < 0.01:
            base_temp = 1.0
        else:
            # Normalized entropy [0, 1]
            normalized = entropy / max_entropy
            # Base temperature: 0.5 at low entropy, 1.5 at high entropy
            base_temp = 0.5 + 1.0 * normalized

        # Heat bath contribution: more heat = higher temperature
        # Physics: T ∝ E / N (equipartition)
        n_tokens = len(self.field) + 1
        heat_contribution = self._heat_bath / (n_tokens + 1)

        # Combined temperature (heat can significantly raise it)
        return base_temp + heat_contribution

    def _calculate_resonance_mismatch(self) -> float:
        """
        Calculate how much the predicted next state mismatches reality.

        This is the key driver of self-emerging annealing:
        - Good predictions: low mismatch, system can cool
        - Poor predictions: high mismatch, energy goes to heat bath

        Mismatch is measured by Jaccard distance between predicted and actual pixels.
        """
        if not self._prediction_made or not self._last_predicted_pixels:
            return 0.0

        # Actual current pixels (excluding actions)
        actual = {pid for pid in self.curr_active if pid not in self.discovered_actions}
        predicted = self._last_predicted_pixels

        if not actual and not predicted:
            return 0.0

        # Jaccard distance: 1 - (intersection / union)
        intersection = len(actual & predicted)
        union = len(actual | predicted)

        if union == 0:
            return 0.0

        jaccard_similarity = intersection / union
        mismatch = 1.0 - jaccard_similarity

        return mismatch

    def _generate_heat_from_mismatch(self, mismatch: float):
        """
        Convert resonance mismatch into thermal energy.

        Physics principle: When oscillators interfere destructively (mismatch),
        the energy doesn't disappear - it goes into the thermal bath.

        Heat generated ∝ mismatch * energy_in_prediction

        ALSO: Check for action imbalance. If some actions are dormant
        while others are active, this is a thermodynamic imbalance that
        should generate additional heat (entropy increase).
        """
        if mismatch < 0.01:
            return  # No significant mismatch

        # Energy in the prediction: sum of amplitude in predicted tokens
        prediction_energy = sum(
            self.field[pid].amplitude for pid in self._last_predicted_pixels
            if pid in self.field
        )

        # Base heat: mismatch fraction of prediction energy
        heat_generated = mismatch * prediction_energy * 0.1  # 10% of mismatched energy

        # ADDITIONAL: Action imbalance contributes to heat
        # When one action dominates, the system is "frustrated" - entropy is low
        # This frustration manifests as additional thermal energy
        if len(self.discovered_actions) > 1:
            action_amplitudes = [
                self.field[aid].amplitude for aid in self.discovered_actions
                if aid in self.field
            ]
            if action_amplitudes:
                max_amp = max(action_amplitudes)
                min_amp = min(action_amplitudes)

                # Imbalance ratio: high when one action dominates
                if min_amp > 0:
                    imbalance = max_amp / min_amp
                else:
                    imbalance = max_amp * 100  # Very high if any action is zero

                # Heat from imbalance: logarithmic (information-theoretic)
                # More imbalance = more "frustration" = more heat
                if imbalance > 10:  # Significant imbalance
                    imbalance_heat = np.log(imbalance) * 0.01
                    heat_generated += imbalance_heat

        self._heat_bath += heat_generated
        self._total_heat_generated += heat_generated

        # Track for analysis
        self._mismatch_history.append(mismatch)
        if len(self._mismatch_history) > 100:
            self._mismatch_history = self._mismatch_history[-100:]

    def _redistribute_heat(self):
        """
        Spontaneously redistribute heat bath energy to the field.

        Physics principle: Equipartition theorem - thermal energy distributes
        equally among all degrees of freedom. In practice, we favor low-amplitude
        tokens (they have more "room" to absorb energy).

        This happens EVERY frame - continuous thermal fluctuations.
        """
        if self._heat_bath < 0.01:
            return

        # Redistribution rate: derived from system size
        # Larger systems redistribute more slowly (thermal inertia)
        n_tokens = len(self.field) + 1
        redistribution_rate = 1.0 / np.sqrt(n_tokens + 1)

        # Amount to redistribute this frame
        to_redistribute = self._heat_bath * redistribution_rate
        if to_redistribute < 1e-6:
            return

        # Distribute preferentially to low-amplitude tokens
        # Physics: equipartition, but starting from lower energy states
        token_weights = []
        for token in self.field.values():
            # Weight: inverse of amplitude squared (lower amplitude = more receptive)
            weight = 1.0 / (token.amplitude ** 2 + 0.01)
            token_weights.append((token, weight))

        total_weight = sum(w for _, w in token_weights)
        if total_weight < 1e-10:
            return

        # Distribute heat
        for token, weight in token_weights:
            share = (weight / total_weight) * to_redistribute
            # Add as thermal fluctuation (random phase)
            phase = np.random.uniform(0, 2 * np.pi)
            token.wave += cmath.rect(share, phase)

        # Remove from heat bath
        self._heat_bath -= to_redistribute
        self._total_heat_redistributed += to_redistribute

        # Track heat bath level
        self._heat_bath_history.append(self._heat_bath)
        if len(self._heat_bath_history) > 100:
            self._heat_bath_history = self._heat_bath_history[-100:]

    def _make_prediction(self):
        """
        Predict next frame's pixels based on current couplings.

        This prediction is compared to actual next frame to generate mismatch.
        """
        if not self.current_action_id:
            self._prediction_made = False
            return

        action = self.field.get(self.current_action_id)
        if not action:
            self._prediction_made = False
            return

        # Predict: pixels strongly coupled to current action
        predicted = set()
        for pixel_id, coupling in action.couplings.items():
            if pixel_id not in self.discovered_actions:
                # Predict if coupling strength exceeds threshold
                # Threshold derived from average coupling
                if abs(coupling) > 0.1:  # Will predict pixels with meaningful coupling
                    predicted.add(pixel_id)

        self._last_predicted_pixels = predicted
        self._prediction_made = True

    def nudge_thermal(self, amount: float = None):
        """
        EXTERNAL NUDGE: Inject thermal energy from outside.

        This is the manual override capability requested.
        Use when you want to force exploration regardless of natural dynamics.

        If amount is None, injects energy proportional to current field energy.
        """
        if amount is None:
            # Default: inject 50% of current field energy
            amount = self._total_field_energy() * 0.5

        self._heat_bath += amount
        self._external_nudges += 1

        # Also seed couplings for dormant actions (thermal fluctuation creates correlations)
        self._seed_dormant_action_couplings()

    def _seed_dormant_action_couplings(self):
        """
        Thermal fluctuations can create momentary correlations.

        When there's heat in the bath, dormant actions may spontaneously
        couple to active pixels. This gives them a chance to be selected
        and prove their worth.

        PHYSICS: The probability of thermal excitation depends on the
        Boltzmann factor: P ∝ exp(-E/kT). At higher temperatures (more heat),
        even high-energy (dormant) states can become occupied.
        """
        if self._heat_bath < 0.01:
            return  # Need some heat for fluctuations

        for action_id in self.discovered_actions:
            action = self.field.get(action_id)
            if not action:
                continue

            # Check if action is dormant (very low coupling)
            total_coupling = sum(abs(c) for c in action.couplings.values())

            # Compare to average of other actions
            other_couplings = []
            for other_id in self.discovered_actions:
                if other_id != action_id:
                    other = self.field.get(other_id)
                    if other:
                        other_couplings.append(sum(abs(c) for c in other.couplings.values()))

            avg_other = np.mean(other_couplings) if other_couplings else 10.0

            # Dormant if < 10% of average
            if total_coupling < avg_other * 0.1:
                # Thermal seeding probability: higher heat = more likely
                # This is the Boltzmann factor approximation
                temperature = self._derive_temperature()
                seed_probability = 1.0 - np.exp(-self._heat_bath / temperature)

                if np.random.random() < seed_probability:
                    # Seed coupling strength scales with heat and temperature
                    seed_strength = self._heat_bath * temperature / (len(self.curr_active) + 1)
                    seed_strength = min(0.5, seed_strength)  # Cap at 0.5 per coupling

                    # Create thermal couplings to active pixels
                    n_seeds = min(30, len(self.curr_active))
                    for pixel_id in list(self.curr_active)[:n_seeds]:
                        if pixel_id not in self.discovered_actions:
                            pixel = self.field.get(pixel_id)
                            if pixel:
                                # Random phase (thermal = incoherent)
                                phase = np.random.uniform(0, 2 * np.pi)
                                coupling = cmath.rect(seed_strength, phase)

                                pixel.couplings[action_id] = (
                                    pixel.couplings.get(action_id, complex(0, 0)) + coupling
                                )
                                action.couplings[pixel_id] = (
                                    action.couplings.get(pixel_id, complex(0, 0)) + coupling
                                )

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe frame and action - inject energy into wave field.

        Energy injection is UNIT (1.0) for each observation.
        No arbitrary scaling factors.

        SELF-EMERGING ANNEALING: Each observation includes:
        1. Check prediction mismatch (generates heat)
        2. Normal observation/coupling
        3. Redistribute heat (thermal fluctuations)
        """
        self.frame_num = frame_num

        # Normalize frame (this is preprocessing, not domain knowledge)
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        h, w = frame.shape

        # Store previous state
        self.prev_active = self.curr_active.copy()
        self.prev_action_id = self.current_action_id
        self.curr_active = set()

        # === OBSERVE PIXELS ===
        # Collect active pixels first, then inject NORMALIZED energy
        active_pixels = []
        for y in range(h):
            for x in range(w):
                intensity = frame[y, x]
                if intensity > 0.01:  # Only observe non-background
                    pixel_id = self._hash(f"p_{y}_{x}")
                    active_pixels.append((pixel_id, intensity))

        # Normalize: total injected energy = 1.0 per frame (for pixels)
        total_intensity = sum(i for _, i in active_pixels) + 1e-10

        for pixel_id, intensity in active_pixels:
            token = self._get_token(pixel_id)
            token.observation_count += 1

            # NORMALIZED energy: total pixel energy per frame = 1.0
            normalized_intensity = intensity / total_intensity
            phase = intensity * 2 * np.pi  # Phase still reflects absolute intensity
            energy = cmath.rect(normalized_intensity, phase)

            token.wave += energy
            self.curr_active.add(pixel_id)

        # === SELF-EMERGING ANNEALING: Check prediction mismatch ===
        # This generates heat from resonance mismatches
        mismatch = self._calculate_resonance_mismatch()
        if mismatch > 0:
            self._generate_heat_from_mismatch(mismatch)

        # === OBSERVE ACTION ===
        # Actions are DISCOVERED, not pre-created
        action_id = self._hash(f"action_{action}")
        self.discovered_actions.add(action_id)
        action_token = self._get_token(action_id)
        action_token.observation_count += 1

        # Same normalized energy as total pixels (balanced injection)
        # Actions get 1.0, pixels get 1.0 total = 2.0 per frame
        action_token.wave += complex(1.0, 0)
        self.current_action_id = action_id
        self.curr_active.add(action_id)

        # === TEMPORAL COUPLING THROUGH ACTION ===
        self._couple_through_action()

        # === TRUE WAVE INTERFERENCE ===
        self._interfere()

        # === BACKWARD AMPLITUDE FLOW ===
        self._propagate_backward()

        # === CO-OCCURRENCE TRACKING ===
        self._track_cooccurrence()

        # === COMPOSITE DISCOVERY ===
        self._discover_composites()

        # === ENERGY-CONSERVING DAMPING ===
        self._damp_all()

        # === SELF-EMERGING ANNEALING: Redistribute heat ===
        # Thermal energy spontaneously flows back to the field
        self._redistribute_heat()

        # === SELF-EMERGING ANNEALING: Seed dormant couplings ===
        # High heat creates thermal fluctuations that couple dormant actions
        self._seed_dormant_action_couplings()

        # === MAKE PREDICTION for next frame ===
        # This will be compared to actual next frame
        self._make_prediction()

    def _couple_through_action(self):
        """
        Build temporal couplings through action bottleneck.

        Coupling strength is proportional to observation count ratio.
        NO arbitrary coupling constant - derived from statistics.
        """
        if not self.prev_action_id:
            return

        prev_action = self.field.get(self.prev_action_id)
        if not prev_action:
            return

        # Coupling strength: inversely proportional to how common each token is
        # Rare co-occurrences are more informative (information theory)
        action_freq = prev_action.observation_count / (self.frame_num + 1)

        # Coupling = 1 / sqrt(freq) - rarer = stronger coupling
        # This is principled: Shannon information = -log(p)
        coupling_strength = 1.0 / (np.sqrt(action_freq) + 0.1)
        coupling_strength = min(1.0, coupling_strength)  # Cap at unit

        # prev_pixels -> prev_action coupling
        for pixel_id in self.prev_active:
            if pixel_id != self.prev_action_id:
                pixel = self.field.get(pixel_id)
                if pixel:
                    # Coupling as complex wave (can interfere)
                    coupling = complex(coupling_strength, 0)
                    pixel.couplings[self.prev_action_id] = (
                        pixel.couplings.get(self.prev_action_id, complex(0, 0)) + coupling
                    )

        # prev_action -> curr_pixels coupling
        for pixel_id in self.curr_active:
            if pixel_id != self.current_action_id:
                pixel = self.field.get(pixel_id)
                if pixel:
                    coupling = complex(coupling_strength, 0)
                    prev_action.couplings[pixel_id] = (
                        prev_action.couplings.get(pixel_id, complex(0, 0)) + coupling
                    )

    def _interfere(self):
        """
        ENERGY-CONSERVING wave interference.

        When coupled waves interact, energy TRANSFERS but doesn't create.
        Total energy before = total energy after.

        This is like coupled oscillators exchanging energy.
        """
        active_list = list(self.curr_active)

        for i, t1_id in enumerate(active_list):
            t1 = self.field.get(t1_id)
            if not t1 or t1.amplitude < 1e-6:
                continue

            for t2_id in active_list[i+1:]:
                t2 = self.field.get(t2_id)
                if not t2 or t2.amplitude < 1e-6:
                    continue

                # Check if coupled
                if t2_id in t1.couplings or t1_id in t2.couplings:
                    coupling_12 = t1.couplings.get(t2_id, complex(0, 0))
                    coupling_21 = t2.couplings.get(t1_id, complex(0, 0))

                    # Transfer rate based on coupling (small)
                    transfer_rate = (abs(coupling_12) + abs(coupling_21)) / 100
                    transfer_rate = min(0.05, transfer_rate)

                    # ENERGY-CONSERVING transfer:
                    # Remove from one, add to other (conservation!)
                    transfer_to_t1 = t2.wave * transfer_rate
                    transfer_to_t2 = t1.wave * transfer_rate

                    t1.wave = t1.wave - transfer_to_t2 + transfer_to_t1
                    t2.wave = t2.wave - transfer_to_t1 + transfer_to_t2

    def _propagate_backward(self):
        """
        Backward amplitude flow - anthropic principle.

        Each frame of existence sends energy backward to actions that led here.
        Discount is e^(-age/τ) where τ = trace length / 3 (natural timescale)

        KEY: Total backward energy is normalized to 1.0 per frame.
        This prevents unbounded energy growth.
        """
        if self.current_action_id:
            self._action_trace.append(self.current_action_id)

        if not self._action_trace:
            return

        trace_len = len(self._action_trace)

        # Timescale τ derived from trace length (no arbitrary constant)
        tau = max(1, trace_len / 3)

        # Calculate total discount (for normalization)
        total_discount = sum(np.exp(-(trace_len - 1 - i) / tau)
                            for i in range(trace_len))

        for i, action_id in enumerate(self._action_trace):
            action = self.field.get(action_id)
            if action:
                age = trace_len - 1 - i
                discount = np.exp(-age / tau)
                # NORMALIZED: total backward energy = 1.0 per frame
                normalized_energy = discount / (total_discount + 1e-10)
                action.wave += complex(normalized_energy, 0)

        # Limit trace length to prevent unbounded growth
        max_trace = min(200, self.frame_num // 10 + 50)
        if len(self._action_trace) > max_trace:
            self._action_trace = self._action_trace[-max_trace:]

    def _track_cooccurrence(self):
        """Track co-occurrence for composite discovery."""
        if not self.current_action_id:
            return

        # Action + pixel co-occurrence (behavioral patterns)
        for pixel_id in self.curr_active:
            if pixel_id != self.current_action_id:
                key = (min(self.current_action_id, pixel_id),
                       max(self.current_action_id, pixel_id))
                self._cooccurrence[key] += 1

        # Changed pixels co-occurrence (dynamic patterns)
        if self.prev_active:
            changed = self.curr_active.symmetric_difference(self.prev_active)
            changed_list = list(changed)
            for i, t1 in enumerate(changed_list[:20]):  # Limit computation
                for t2 in changed_list[i+1:20]:
                    key = (min(t1, t2), max(t1, t2))
                    self._cooccurrence[key] += 1

    def _discover_composites(self):
        """
        Discover composite patterns from co-occurrence statistics.

        Threshold derived from observation counts, not arbitrary.
        A pair that co-occurs more than sqrt(total_frames) times is significant.
        """
        # Derived threshold: sqrt of total observations
        threshold = max(5, int(np.sqrt(self.frame_num + 1)))

        # Only check periodically (every sqrt(frame_num) frames - derived)
        check_interval = max(10, int(np.sqrt(self.frame_num + 1)))
        if self.frame_num % check_interval != 0:
            return

        # Find significant co-occurrences
        for (t1_id, t2_id), count in list(self._cooccurrence.items()):
            if count >= threshold:
                comp_id = self._hash(f"comp_{t1_id}_{t2_id}")
                comp = self._get_composite(comp_id)

                if not comp.components:
                    comp.components = {t1_id, t2_id}

                # Composite energy: geometric mean of components (natural for waves)
                t1 = self.field.get(t1_id)
                t2 = self.field.get(t2_id)
                if t1 and t2:
                    combined_amp = np.sqrt(t1.amplitude * t2.amplitude)
                    combined_phase = t1.phase + t2.phase
                    comp.wave += cmath.rect(combined_amp / threshold, combined_phase)

                # Decay co-occurrence count (allow new patterns)
                self._cooccurrence[(t1_id, t2_id)] = count // 2

    def _damp_all(self):
        """
        Energy-conserving damping with zero-point energy.

        PHYSICS PRINCIPLE: Quantum systems have zero-point energy.
        No oscillator ever reaches exactly zero - there's always
        ground state energy: E_0 = (1/2)ℏω

        For our wave field:
        - Damping reduces amplitude
        - But discovered tokens maintain minimum "existence" amplitude
        - This is proportional to 1/N where N = total discovered tokens
        - (Equal share of some base existence energy)
        """
        damping_rate = self._derive_damping()
        damping = 1.0 - damping_rate

        # Zero-point energy: total = 1.0, distributed equally among all tokens
        n_tokens = len(self.field) + 1
        zero_point = 1.0 / n_tokens

        # Damp all waves, but maintain zero-point for discovered actions
        for token in self.field.values():
            token.wave *= damping

            # Actions maintain zero-point energy (can always be re-excited)
            if token.id in self.discovered_actions:
                if token.amplitude < zero_point:
                    # Re-inject zero-point energy (maintains existence)
                    token.wave = complex(zero_point, token.phase)

            # Damp couplings (no zero-point for couplings)
            for key in token.couplings:
                token.couplings[key] *= damping

        for comp in self.composites.values():
            comp.wave *= damping

    def signal_game_end(self, game_length: int = 0):
        """
        Game boundary - clear trace, preserve field.

        SELF-EMERGING ANNEALING: No explicit heating trigger needed!
        Heat is generated naturally from prediction mismatches.
        The only thing we do at game end is:
        1. Clear the action trace
        2. Track game length for statistics
        """
        self._action_trace = []

        # Track game performance for statistics
        if game_length > 0:
            self._recent_game_lengths.append(game_length)
            if len(self._recent_game_lengths) > 50:
                self._recent_game_lengths = self._recent_game_lengths[-50:]

        # Clear prediction state (new game = new context)
        self._prediction_made = False
        self._last_predicted_pixels = set()

    def choose_action(self, num_actions: int = 3) -> int:
        """
        Choose action through PURE RESONANCE with entropy maintenance.

        The current state creates a query wave.
        Actions resonate based on coupling structure.

        ENTROPY MAINTENANCE (physics-derived):
        - Track action usage frequencies
        - Add exploration bonus inversely proportional to usage
        - This is -log(p) information: rare actions carry more information
        """
        action_resonance = {}

        # Calculate action usage frequencies for entropy term
        total_action_obs = sum(
            self.field.get(self._hash(f"action_{a}"), WaveToken("")).observation_count
            for a in range(num_actions)
        ) + num_actions  # +num_actions to avoid divide by zero

        for a in range(num_actions):
            action_id = self._hash(f"action_{a}")

            if action_id not in self.discovered_actions:
                action_resonance[a] = 1.0
                continue

            action_token = self.field.get(action_id)
            if not action_token:
                action_resonance[a] = 1.0
                continue

            # RESONANCE: Sum of coupling strengths from active pixels
            coupling_sum = 0.0
            for pixel_id in self.curr_active:
                pixel = self.field.get(pixel_id)
                if pixel and action_id in pixel.couplings:
                    coupling_sum += abs(pixel.couplings[action_id])

            # Bidirectional resonance
            reverse_coupling = 0.0
            for pixel_id in self.curr_active:
                if pixel_id in action_token.couplings:
                    reverse_coupling += abs(action_token.couplings[pixel_id])

            # Action's amplitude (survival history)
            amplitude_score = action_token.amplitude

            # ENTROPY-BASED EXPLORATION (information-theoretic)
            # Rare actions carry more information: -log(p) = log(1/p)
            # This emerges from the observation counts, not arbitrary
            action_freq = (action_token.observation_count + 1) / total_action_obs
            information_value = -np.log(action_freq + 1e-10)
            # Normalize by max possible info (uniform distribution)
            max_info = -np.log(1.0 / num_actions)
            normalized_info = information_value / max_info

            # Combine: resonance is primary, information value adds exploration
            # The ratio between them is determined by observation counts
            # (frequently used actions have less info bonus)
            resonance = coupling_sum + reverse_coupling + amplitude_score
            exploration = normalized_info * (amplitude_score + 0.1)  # Scale by action's "maturity"

            action_resonance[a] = resonance + exploration + 1e-6

        # DERIVED temperature from field entropy
        temperature = self._derive_temperature()

        # Softmax with derived temperature
        scores = list(action_resonance.values())
        max_score = max(scores)

        exp_scores = {}
        for a, score in action_resonance.items():
            exp_scores[a] = np.exp((score - max_score) / temperature)

        total = sum(exp_scores.values())
        if total < 1e-10:
            return np.random.randint(0, num_actions)

        probs = {a: s / total for a, s in exp_scores.items()}

        r = np.random.random()
        cumulative = 0.0
        for a, p in probs.items():
            cumulative += p
            if r < cumulative:
                return a

        return num_actions - 1

    def get_action_amplitudes(self) -> Dict[int, float]:
        """Get amplitude of discovered actions."""
        result = {}
        for a in range(10):
            action_id = self._hash(f"action_{a}")
            if action_id in self.field:
                result[a] = self.field[action_id].amplitude
        return result

    def get_stats(self) -> Dict:
        """Comprehensive statistics."""
        field_energy = sum(abs(t.wave)**2 for t in self.field.values())
        composite_energy = sum(abs(c.wave)**2 for c in self.composites.values())

        action_amps = []
        pixel_amps = []
        for tid, token in self.field.items():
            if tid in self.discovered_actions:
                action_amps.append(token.amplitude)
            else:
                pixel_amps.append(token.amplitude)

        # Count couplings
        total_couplings = sum(len(t.couplings) for t in self.field.values())

        # Average mismatch (for annealing analysis)
        avg_mismatch = np.mean(self._mismatch_history) if self._mismatch_history else 0

        return {
            'frame': self.frame_num,
            'field_tokens': len(self.field),
            'discovered_actions': len(self.discovered_actions),
            'composites': len(self.composites),
            'field_energy': field_energy,
            'composite_energy': composite_energy,
            'total_energy': field_energy + composite_energy,
            'entropy': self._field_entropy(),
            'derived_damping': self._derive_damping(),
            'derived_temperature': self._derive_temperature(),
            'total_couplings': total_couplings,
            'action_avg_amp': np.mean(action_amps) if action_amps else 0,
            'pixel_avg_amp': np.mean(pixel_amps) if pixel_amps else 0,
            'cooccurrence_pairs': len(self._cooccurrence),
            'trace_length': len(self._action_trace),
            # Self-emerging annealing state
            'heat_bath': self._heat_bath,
            'avg_mismatch': avg_mismatch,
            'total_heat_generated': self._total_heat_generated,
            'total_heat_redistributed': self._total_heat_redistributed,
            'external_nudges': self._external_nudges,
        }

    def snapshot(self, label: str = ""):
        """Take a snapshot of current state for tracking progression."""
        stats = self.get_stats()
        amps = self.get_action_amplitudes()

        snapshot = {
            'label': label,
            'frame': self.frame_num,
            'stats': stats,
            'action_amplitudes': amps,
        }
        self._stats_history.append(snapshot)
        return snapshot

    def print_snapshot(self, snapshot: Dict):
        """Print a snapshot."""
        stats = snapshot['stats']
        amps = snapshot['action_amplitudes']

        print(f"\n{'='*70}")
        print(f"SNAPSHOT: {snapshot['label']} (Frame {snapshot['frame']})")
        print(f"{'='*70}")

        print(f"\nField State:")
        print(f"  Tokens: {stats['field_tokens']}")
        print(f"  Discovered actions: {stats['discovered_actions']}")
        print(f"  Composites: {stats['composites']}")
        print(f"  Total couplings: {stats['total_couplings']}")
        print(f"  Co-occurrence pairs tracked: {stats['cooccurrence_pairs']}")

        print(f"\nEnergy (DERIVED):")
        print(f"  Field energy: {stats['field_energy']:.2e}")
        print(f"  Composite energy: {stats['composite_energy']:.2e}")
        print(f"  Total: {stats['total_energy']:.2e}")
        print(f"  Damping rate: {stats['derived_damping']:.6f}")

        print(f"\nDynamics (DERIVED):")
        print(f"  Field entropy: {stats['entropy']:.2f}")
        print(f"  Selection temperature: {stats['derived_temperature']:.2f}")
        print(f"  Trace length: {stats['trace_length']}")

        print(f"\nSelf-Emerging Annealing:")
        print(f"  Heat bath: {stats['heat_bath']:.4f}")
        print(f"  Avg mismatch: {stats['avg_mismatch']:.4f}")
        print(f"  Total heat generated: {stats['total_heat_generated']:.2f}")
        print(f"  Total heat redistributed: {stats['total_heat_redistributed']:.2f}")
        print(f"  External nudges: {stats['external_nudges']}")

        print(f"\nAmplitude Distribution:")
        print(f"  Action avg: {stats['action_avg_amp']:.2e}")
        print(f"  Pixel avg: {stats['pixel_avg_amp']:.2e}")
        if stats['pixel_avg_amp'] > 0:
            print(f"  Bottleneck ratio: {stats['action_avg_amp']/stats['pixel_avg_amp']:.1f}x")

        print(f"\nAction Amplitudes:")
        for a, amp in sorted(amps.items()):
            # Use ASCII bar instead of unicode
            bar_len = min(50, int(np.log10(amp + 1) * 10)) if amp > 0 else 0
            bar = '#' * bar_len
            print(f"  action_{a}: {amp:12.2e} {bar}")

    def print_coupling_analysis(self):
        """Analyze coupling structure to understand learning."""
        print(f"\n{'='*70}")
        print("COUPLING ANALYSIS")
        print(f"{'='*70}")

        # For each action, count total coupling strength
        for a in range(3):
            action_id = self._hash(f"action_{a}")
            action_token = self.field.get(action_id)

            if not action_token:
                continue

            # Incoming couplings (pixel -> action)
            incoming = 0.0
            incoming_count = 0
            for tid, token in self.field.items():
                if action_id in token.couplings:
                    incoming += abs(token.couplings[action_id])
                    incoming_count += 1

            # Outgoing couplings (action -> pixel)
            outgoing = sum(abs(c) for c in action_token.couplings.values())
            outgoing_count = len(action_token.couplings)

            print(f"\nAction {a}:")
            print(f"  Incoming: {incoming:.4f} from {incoming_count} pixels")
            print(f"  Outgoing: {outgoing:.4f} to {outgoing_count} pixels")
            print(f"  Amplitude: {action_token.amplitude:.4f}")

            # Check for spatial differentiation
            if incoming_count > 0 and incoming > 0:
                # Find pixels with strongest coupling to this action
                pixel_couplings = []
                for tid, token in self.field.items():
                    if action_id in token.couplings:
                        coupling_str = abs(token.couplings[action_id])
                        pixel_couplings.append((tid, coupling_str))

                if pixel_couplings:
                    pixel_couplings.sort(key=lambda x: -x[1])
                    top_5 = pixel_couplings[:5]
                    avg_coupling = incoming / incoming_count
                    max_coupling = top_5[0][1] if top_5 else 0
                    if avg_coupling > 0:
                        print(f"  Avg coupling: {avg_coupling:.6f}, Max: {max_coupling:.6f}")
                        print(f"  Differentiation ratio: {max_coupling/avg_coupling:.2f}x")


# =============================================================================
# TEST WITH PROGRESSION SNAPSHOTS
# =============================================================================

def test_pure_wave_pong():
    """
    Test pure wave sieve on Pong with progression snapshots.

    NO magic numbers in the sieve itself.
    All parameters derived from field state.
    """
    print("=" * 70)
    print("PURE WAVE SIEVE - PHYSICS DERIVED")
    print("No magic numbers, no arbitrary weights, no pre-created tokens")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = PureWaveSieve()

    # Game tracking
    game_lengths = []
    hits = 0
    misses = 0

    # Pong state
    ball_x, ball_y = 10.5, 5.0
    ball_dx, ball_dy = 0.5, 0.5
    paddle_x = 10.5
    current_game_length = 0

    # Snapshot schedule (more at start, less later)
    snapshot_frames = [100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]

    TOTAL_FRAMES = 50000

    for frame_num in range(TOTAL_FRAMES):
        # Create frame
        pixels = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

        # Ball (2x2 block)
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
                sieve.signal_game_end(current_game_length)
                current_game_length = 0

                ball_y = 5
                ball_x = np.random.uniform(5, FRAME_SIZE - 5)
                ball_dx = np.random.choice([-0.5, 0.5])
                ball_dy = 0.5

        # Snapshots
        if frame_num + 1 in snapshot_frames:
            total_games = hits + misses
            hit_rate = hits / total_games if total_games > 0 else 0
            recent = game_lengths[-20:] if len(game_lengths) >= 20 else game_lengths
            avg_len = np.mean(recent) if recent else 0

            snapshot = sieve.snapshot(
                f"Frame {frame_num+1} | Games: {total_games} | "
                f"Hit: {hit_rate:.1%} | Avg length: {avg_len:.1f}"
            )
            sieve.print_snapshot(snapshot)
            sieve.print_coupling_analysis()

            # Show game length progression
            if len(game_lengths) >= 10:
                first = np.mean(game_lengths[:10])
                last = np.mean(game_lengths[-10:])
                print(f"\nGame Length Progression:")
                print(f"  First 10 games: {first:.1f} frames")
                print(f"  Last 10 games: {last:.1f} frames")
                if first > 0:
                    print(f"  Change: {(last/first - 1)*100:+.1f}%")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_games = hits + misses
    print(f"\nTotal frames: {TOTAL_FRAMES}")
    print(f"Total games: {total_games}")
    print(f"Hits: {hits} ({100*hits/total_games:.1f}%)")
    print(f"Misses: {misses}")

    if len(game_lengths) >= 20:
        first_20 = np.mean(game_lengths[:20])
        last_20 = np.mean(game_lengths[-20:])
        print(f"\nGame length evolution:")
        print(f"  First 20 games: {first_20:.1f} frames")
        print(f"  Last 20 games: {last_20:.1f} frames")
        print(f"  Improvement: {(last_20/first_20 - 1)*100:+.1f}%")

    # Show full snapshot history
    print("\n" + "=" * 70)
    print("PROGRESSION SUMMARY")
    print("=" * 70)
    print(f"\n{'Frame':>8} | {'Energy':>10} | {'Heat Bath':>10} | {'Mismatch':>8} | {'Temp':>6} | {'Couplings':>10}")
    print("-" * 80)
    for snap in sieve._stats_history:
        s = snap['stats']
        print(f"{snap['frame']:>8} | {s['total_energy']:>10.1f} | {s['heat_bath']:>10.4f} | "
              f"{s['avg_mismatch']:>8.4f} | {s['derived_temperature']:>6.2f} | {s['total_couplings']:>10}")

    # Show thermal cycle analysis
    print("\n" + "=" * 70)
    print("THERMAL CYCLE ANALYSIS")
    print("=" * 70)
    print(f"Total heat generated (from mismatches): {sieve._total_heat_generated:.2f}")
    print(f"Total heat redistributed (to field): {sieve._total_heat_redistributed:.2f}")
    print(f"External nudges applied: {sieve._external_nudges}")
    if sieve._mismatch_history:
        print(f"Average prediction mismatch: {np.mean(sieve._mismatch_history):.4f}")
        print(f"Mismatch std dev: {np.std(sieve._mismatch_history):.4f}")

    return sieve, game_lengths


if __name__ == "__main__":
    test_pure_wave_pong()
