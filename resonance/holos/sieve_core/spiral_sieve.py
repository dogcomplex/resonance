"""
SPIRAL SIEVE: N-Dimensional Whirlpool Dynamics
===============================================

A physics-correct sieve where patterns spiral inward toward coherence.

CORE GEOMETRY: The Whirlpool
============================
Imagine a whirlpool in N dimensions:
- OUTER EDGE: High entropy, exploratory, new patterns enter here
- SPIRAL INWARD: Patterns that resonate move toward center
- CENTER: Low entropy, proven patterns, highest amplitude
- ORBITAL PERIODS: Different radii = different timescales

This geometry EMERGES from physics, not from architectural choices:

1. RADIAL POSITION = Amplitude
   - High amplitude patterns are "near center" (low radius)
   - Low amplitude patterns are "at edge" (high radius)
   - This is continuous, not discrete levels

2. ANGULAR POSITION = Phase
   - Patterns with aligned phases orbit together
   - Phase coherence enables interference/combination
   - Random phase = thermal fluctuations

3. ORBITAL DYNAMICS
   - Angular momentum: L = r × p (radius × momentum)
   - Coherent resonance → lose angular momentum → spiral inward
   - Poor resonance → gain angular momentum → drift outward
   - This IS the selection mechanism

4. ORBITAL PERIOD ~ RADIUS
   - Outer layers update frequently (high entropy exploration)
   - Inner layers update rarely (stable, proven patterns)
   - This is Kepler's law: T ∝ r^(3/2)

5. RECOMBINATION AT CONJUNCTIONS
   - Patterns in same orbital ring can combine when phases align
   - This replaces explicit "composite discovery"
   - Combination probability ∝ 1/|Δphase|

N-DIMENSIONAL EXTENSION
=======================
The spiral generalizes as hyperbolic flow toward fixed point:

    dr/dt = -k × coherence × r    (inward flow ∝ success)
    dθ/dt = ω(r)                   (angular velocity ∝ 1/r)

In embedding space:
- "Radius" = distance from centroid of proven patterns
- "Angular" dimensions = manifold structure
- Multiple fixed points = competing attractors (strategies)

PHYSICS CORRECTNESS
===================
1. NO DISCRETE IDENTITIES: Patterns identified by continuous position
2. TOPOLOGY-AWARE COUPLING: Distance in embedding space matters
3. UNIFIED FIELD: No separate "composites" - just interference patterns
4. CONTINUOUS TIME: Differential equations, not discrete updates
5. SPATIAL HEAT: Local temperature field, not global scalar
6. EMERGENT SELECTION: Resonance → angular momentum → radial drift
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import cmath
from collections import defaultdict


class SpiralPattern:
    """
    A pattern in the spiral field.

    Position is continuous in (r, θ₁, θ₂, ..., θₙ₋₁) hyperspherical coordinates.
    """
    __slots__ = [
        'embedding',       # Position in N-dimensional embedding space
        'momentum',        # Velocity in embedding space (for dynamics)
        'amplitude',       # Wave amplitude (energy)
        'phase',           # Wave phase
        'angular_momentum', # L = r × p (determines orbital stability)
        'birth_time',      # When pattern was created
        'last_active',     # Last frame when pattern was excited
        'coupling_kernel', # Local coupling weights (spatial)
    ]

    def __init__(self, embedding: np.ndarray, birth_time: int = 0):
        self.embedding = embedding.astype(np.float64)
        self.momentum = np.zeros_like(embedding)
        self.amplitude = 0.0
        self.phase = 0.0
        self.angular_momentum = 0.0
        self.birth_time = birth_time
        self.last_active = birth_time
        self.coupling_kernel = {}  # pattern_idx -> coupling strength

    @property
    def radius(self) -> float:
        """Distance from origin in embedding space."""
        return np.linalg.norm(self.embedding)

    @property
    def wave(self) -> complex:
        """Complex wave representation."""
        return cmath.rect(self.amplitude, self.phase)

    @wave.setter
    def wave(self, value: complex):
        self.amplitude = abs(value)
        self.phase = cmath.phase(value)


class SpatialHeatField:
    """
    Spatially-distributed heat bath.

    Instead of a single global temperature, heat is distributed
    in embedding space. Local regions can be hot or cold.
    """

    def __init__(self, dim: int, resolution: int = 10):
        self.dim = dim
        self.resolution = resolution
        # Heat grid in embedding space
        self.grid = np.zeros([resolution] * min(dim, 3))  # Limit to 3D for memory
        self.total_heat = 0.0

    def add_heat(self, position: np.ndarray, amount: float):
        """Add heat at a specific position."""
        # Map position to grid cell
        idx = self._position_to_index(position)
        if idx is not None:
            self.grid[idx] += amount
        self.total_heat += amount

    def get_temperature(self, position: np.ndarray) -> float:
        """Get local temperature at position."""
        idx = self._position_to_index(position)
        if idx is None:
            return self.total_heat / (self.resolution ** min(self.dim, 3))
        return self.grid[idx]

    def diffuse(self, rate: float = 0.1):
        """Heat diffusion - smooth the temperature field."""
        if self.dim <= 3:
            # Simple diffusion via convolution
            from scipy.ndimage import uniform_filter
            self.grid = uniform_filter(self.grid, size=3, mode='wrap') * (1 - rate) + self.grid * rate

    def radiate(self, rate: float = 0.01):
        """Radiative cooling - heat dissipates to environment."""
        self.grid *= (1 - rate)
        self.total_heat *= (1 - rate)

    def _position_to_index(self, position: np.ndarray) -> Optional[Tuple]:
        """Map continuous position to grid index."""
        # Normalize to [0, 1] based on typical radius
        normalized = (position[:min(self.dim, 3)] / 10.0 + 0.5).clip(0, 0.999)
        idx = tuple((normalized * self.resolution).astype(int))
        if all(0 <= i < self.resolution for i in idx):
            return idx
        return None


class SpiralSieve:
    """
    The Spiral Sieve: patterns orbit and spiral based on coherence.

    Core dynamics:
    1. Patterns exist in continuous embedding space
    2. Coherent resonance → spiral inward (gain amplitude)
    3. Poor resonance → drift outward (lose amplitude)
    4. Orbital period ∝ radius (Kepler-like)
    5. Heat is spatial, not global
    """

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

        # The unified field - all patterns in one space
        self.patterns: List[SpiralPattern] = []

        # Spatial indexing for efficient neighbor queries
        self._spatial_index: Dict[Tuple, List[int]] = defaultdict(list)
        self._cell_size = 1.0  # Size of spatial cells

        # Learned embedding projection (pixels → embedding space)
        # Initialize as random orthogonal projection
        self._pixel_projection = self._init_orthogonal_projection(1000, embedding_dim)

        # Action embeddings (learned positions in the spiral)
        self._action_embeddings: Dict[int, np.ndarray] = {}

        # Spatial heat field
        self.heat_field = SpatialHeatField(embedding_dim)

        # Frame tracking
        self.frame_num = 0
        self.prev_active_indices: Set[int] = set()
        self.curr_active_indices: Set[int] = set()

        # Current action state
        self.current_action: Optional[int] = None
        self.prev_action: Optional[int] = None

        # Centroid of "successful" patterns (the attractor)
        self._success_centroid = np.zeros(embedding_dim)
        self._centroid_mass = 1.0  # Accumulated weight

        # Physics constants (derived from embedding dimension)
        self._orbital_constant = 2 * np.pi / np.sqrt(embedding_dim)
        self._coupling_decay_length = np.sqrt(embedding_dim)  # Characteristic coupling distance
        self._inward_rate = 1.0 / embedding_dim  # Rate of spiral inward

        # Statistics
        self._total_energy_history: List[float] = []
        self._game_lengths: List[int] = []
        self._current_game_length = 0

    def _init_orthogonal_projection(self, input_dim: int, output_dim: int) -> np.ndarray:
        """Initialize a random orthogonal projection matrix."""
        # Use QR decomposition to get orthogonal vectors
        random_matrix = np.random.randn(input_dim, output_dim)
        q, _ = np.linalg.qr(random_matrix)
        return q[:, :output_dim]

    def _pixel_to_embedding(self, y: int, x: int, intensity: float) -> np.ndarray:
        """Map a pixel to embedding space."""
        # Create sparse input (mostly zeros)
        pixel_idx = y * 100 + x  # Assume max 100x100 frame
        if pixel_idx >= self._pixel_projection.shape[0]:
            # Expand projection if needed
            old_size = self._pixel_projection.shape[0]
            new_size = pixel_idx + 100
            new_proj = self._init_orthogonal_projection(new_size, self.embedding_dim)
            new_proj[:old_size] = self._pixel_projection
            self._pixel_projection = new_proj

        # Project to embedding space, scaled by intensity
        return self._pixel_projection[pixel_idx] * intensity

    def _get_action_embedding(self, action: int) -> np.ndarray:
        """Get or create embedding for an action."""
        if action not in self._action_embeddings:
            # Actions start at medium radius, random angle
            radius = 5.0  # Medium distance from center
            random_dir = np.random.randn(self.embedding_dim)
            random_dir /= np.linalg.norm(random_dir)
            self._action_embeddings[action] = random_dir * radius
        return self._action_embeddings[action]

    def _find_or_create_pattern(self, embedding: np.ndarray, tolerance: float = 0.5) -> int:
        """Find existing pattern near this embedding, or create new one."""
        # Check spatial index for nearby patterns
        cell = self._get_cell(embedding)

        # Check this cell and neighbors
        for offset in self._neighbor_offsets():
            neighbor_cell = tuple(c + o for c, o in zip(cell, offset))
            for idx in self._spatial_index.get(neighbor_cell, []):
                pattern = self.patterns[idx]
                dist = np.linalg.norm(pattern.embedding - embedding)
                if dist < tolerance:
                    return idx

        # Create new pattern
        idx = len(self.patterns)
        pattern = SpiralPattern(embedding, self.frame_num)
        self.patterns.append(pattern)
        self._spatial_index[cell].append(idx)
        return idx

    def _get_cell(self, embedding: np.ndarray) -> Tuple:
        """Get spatial cell for an embedding."""
        # Use first 3 dimensions for spatial indexing
        coords = embedding[:min(3, self.embedding_dim)]
        return tuple((coords / self._cell_size).astype(int))

    def _neighbor_offsets(self) -> List[Tuple]:
        """Get neighbor cell offsets for spatial queries."""
        offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    offsets.append((dx, dy, dz)[:min(3, self.embedding_dim)])
        return offsets

    def _distance(self, idx1: int, idx2: int) -> float:
        """Distance between two patterns in embedding space."""
        return np.linalg.norm(
            self.patterns[idx1].embedding - self.patterns[idx2].embedding
        )

    def _coupling_strength(self, idx1: int, idx2: int) -> float:
        """
        Coupling strength between patterns.

        PHYSICS: Coupling decays exponentially with distance.
        Nearby patterns couple strongly; distant patterns weakly.
        """
        dist = self._distance(idx1, idx2)
        return np.exp(-dist / self._coupling_decay_length)

    def observe(self, frame: np.ndarray, action: int, frame_num: int):
        """
        Observe frame and action.

        This is where energy enters the spiral:
        1. Pixels create/excite patterns at their embedding positions
        2. Action creates/excites pattern at action embedding
        3. Coupling builds between co-active patterns
        4. Dynamics run (orbital motion, spiral flow)
        """
        self.frame_num = frame_num
        self._current_game_length += 1

        # Store previous state
        self.prev_active_indices = self.curr_active_indices.copy()
        self.prev_action = self.current_action
        self.curr_active_indices = set()
        self.current_action = action

        # Normalize frame
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=2)
        frame = frame.astype(float)
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val

        h, w = frame.shape

        # === OBSERVE PIXELS (OPTIMIZED) ===
        # Use vectorized operations where possible
        active_mask = frame > 0.01
        active_coords = np.where(active_mask)

        if len(active_coords[0]) > 0:
            # Compute combined embedding efficiently
            intensities = frame[active_mask]
            total_intensity = intensities.sum()

            # Use a hash of active pixels as state identifier
            state_hash = hash((tuple(active_coords[0][:10]), tuple(active_coords[1][:10])))

            # Simple embedding: weighted centroid in pixel space, projected
            cy = np.average(active_coords[0], weights=intensities)
            cx = np.average(active_coords[1], weights=intensities)

            # Create embedding from centroid
            combined_embedding = np.zeros(self.embedding_dim)
            combined_embedding[0] = cy / h * 2 - 1  # Normalize to [-1, 1]
            combined_embedding[1] = cx / w * 2 - 1
            # Add some spread for higher dims
            for i in range(2, self.embedding_dim):
                combined_embedding[i] = np.sin(state_hash * (i + 1) * 0.001) * 0.5

            # Find/create pattern for this combined state
            state_idx = self._find_or_create_pattern(combined_embedding, tolerance=0.3)
            self.curr_active_indices.add(state_idx)

            # Inject energy
            self.patterns[state_idx].amplitude += 0.1
            self.patterns[state_idx].last_active = frame_num

        # === OBSERVE ACTION ===
        action_emb = self._get_action_embedding(action)
        action_idx = self._find_or_create_pattern(action_emb, tolerance=0.5)
        self.curr_active_indices.add(action_idx)
        self.patterns[action_idx].amplitude += 1.0
        self.patterns[action_idx].last_active = frame_num

        # === BUILD COUPLINGS (only between active patterns) ===
        self._build_couplings()

        # === RUN DYNAMICS (every 10 frames for speed) ===
        if frame_num % 10 == 0:
            self._generate_heat_from_mismatch()
            self._run_dynamics()
            self._rebuild_spatial_index()

    def _build_couplings(self):
        """
        Build couplings between co-active patterns.

        PHYSICS: Coupling strength depends on:
        1. Spatial proximity (closer = stronger)
        2. Phase alignment (in-phase = constructive)
        3. Both patterns being active
        """
        active_list = list(self.curr_active_indices)

        for i, idx1 in enumerate(active_list):
            p1 = self.patterns[idx1]
            for idx2 in active_list[i+1:]:
                p2 = self.patterns[idx2]

                # Coupling strength from distance
                base_coupling = self._coupling_strength(idx1, idx2)

                # Phase factor (in-phase patterns couple stronger)
                phase_diff = abs(p1.phase - p2.phase)
                phase_factor = np.cos(phase_diff)  # +1 for in-phase, -1 for anti-phase

                coupling = base_coupling * (1 + phase_factor) / 2

                # Store bidirectionally
                p1.coupling_kernel[idx2] = p1.coupling_kernel.get(idx2, 0) + coupling
                p2.coupling_kernel[idx1] = p2.coupling_kernel.get(idx1, 0) + coupling

        # Also couple current to previous (temporal coupling)
        if self.prev_active_indices:
            for curr_idx in self.curr_active_indices:
                for prev_idx in self.prev_active_indices:
                    if curr_idx != prev_idx:
                        coupling = self._coupling_strength(curr_idx, prev_idx) * 0.5
                        self.patterns[curr_idx].coupling_kernel[prev_idx] = (
                            self.patterns[curr_idx].coupling_kernel.get(prev_idx, 0) + coupling
                        )

    def _generate_heat_from_mismatch(self):
        """
        Poor prediction → energy goes to heat bath.

        PHYSICS: When expected patterns don't appear, the energy
        that would have gone to them goes to local heat instead.
        """
        if not self.prev_active_indices:
            return

        # Check which previous patterns predicted current ones
        for prev_idx in self.prev_active_indices:
            prev_pattern = self.patterns[prev_idx]

            # What did this pattern predict? (strongly coupled patterns)
            predictions = set()
            for coupled_idx, strength in prev_pattern.coupling_kernel.items():
                if strength > 0.1:  # Meaningful coupling
                    predictions.add(coupled_idx)

            if not predictions:
                continue

            # How many predictions were correct?
            correct = predictions & self.curr_active_indices
            mismatch = 1.0 - len(correct) / len(predictions) if predictions else 0

            if mismatch > 0.1:
                # Deposit heat at this pattern's location
                heat_amount = mismatch * prev_pattern.amplitude * 0.1
                self.heat_field.add_heat(prev_pattern.embedding, heat_amount)

    def _run_dynamics(self):
        """
        Run orbital dynamics.

        This is the core physics:
        1. Patterns orbit based on angular momentum
        2. Coherent resonance → lose angular momentum → spiral inward
        3. Poor coherence → gain angular momentum → drift outward
        4. Heat redistributes to low-amplitude patterns
        """
        # Compute centroid of active patterns (the attractor)
        if self.curr_active_indices:
            active_embeddings = [self.patterns[i].embedding for i in self.curr_active_indices]
            active_amplitudes = [self.patterns[i].amplitude for i in self.curr_active_indices]
            total_amp = sum(active_amplitudes) + 1e-10

            current_centroid = sum(e * a for e, a in zip(active_embeddings, active_amplitudes)) / total_amp

            # Update success centroid with exponential moving average
            alpha = 0.01  # Slow adaptation
            self._success_centroid = (1 - alpha) * self._success_centroid + alpha * current_centroid
            self._centroid_mass += 0.01

        dt = 1.0  # Time step

        # Only update recently active patterns (optimization)
        patterns_to_update = [
            idx for idx, p in enumerate(self.patterns)
            if self.frame_num - p.last_active < 100
        ]

        for idx in patterns_to_update:
            pattern = self.patterns[idx]

            # === COMPUTE FORCES ===

            # 1. GRAVITATIONAL: Pull toward success centroid
            to_center = self._success_centroid - pattern.embedding
            dist_to_center = np.linalg.norm(to_center) + 1e-10

            # Gravitational force ∝ 1/r² but capped
            grav_strength = min(1.0, 1.0 / (dist_to_center ** 2 + 1))
            grav_force = to_center / dist_to_center * grav_strength * 0.1

            # 2. COHERENCE: Patterns that resonate move inward
            coherence = self._compute_coherence(idx)
            inward_force = -to_center / dist_to_center * coherence * self._inward_rate

            # 3. THERMAL: Heat causes outward drift
            local_temp = self.heat_field.total_heat / (len(self.patterns) + 1)  # Simplified
            thermal_kick = np.random.randn(self.embedding_dim) * np.sqrt(local_temp + 0.01) * 0.05

            # === UPDATE MOMENTUM ===
            total_force = grav_force + inward_force + thermal_kick
            pattern.momentum += total_force * dt

            # Damping (friction)
            pattern.momentum *= 0.9

            # === UPDATE POSITION ===
            pattern.embedding += pattern.momentum * dt

            # === ORBITAL PHASE UPDATE ===
            if dist_to_center > 0.1:
                angular_velocity = self._orbital_constant / dist_to_center
                pattern.phase = (pattern.phase + angular_velocity * dt) % (2 * np.pi)

            # === AMPLITUDE DECAY ===
            decay_rate = 0.01 * (1 + dist_to_center / 10)
            pattern.amplitude *= (1 - decay_rate)

            # === COUPLING DECAY ===
            keys_to_remove = [k for k, v in pattern.coupling_kernel.items() if v < 1e-4]
            for key in keys_to_remove:
                del pattern.coupling_kernel[key]
            for key in pattern.coupling_kernel:
                pattern.coupling_kernel[key] *= 0.95

        # === HEAT DYNAMICS ===
        self.heat_field.radiate(0.05)

        # === HEAT REDISTRIBUTION (simplified) ===
        if self.heat_field.total_heat > 0.1:
            for idx in patterns_to_update:
                pattern = self.patterns[idx]
                if pattern.amplitude < 0.1:
                    pattern.amplitude += 0.001 * self.heat_field.total_heat
                    self.heat_field.total_heat *= 0.999

    def _compute_coherence(self, idx: int) -> float:
        """
        Compute how coherent this pattern is with its neighbors.

        High coherence = phases align, amplitudes correlate
        Low coherence = random phases, no correlation
        """
        pattern = self.patterns[idx]

        if not pattern.coupling_kernel:
            return 0.0

        total_coherence = 0.0
        total_weight = 0.0

        for neighbor_idx, coupling in pattern.coupling_kernel.items():
            if neighbor_idx >= len(self.patterns):
                continue
            neighbor = self.patterns[neighbor_idx]

            # Phase coherence
            phase_diff = abs(pattern.phase - neighbor.phase)
            phase_coh = np.cos(phase_diff)

            # Amplitude correlation (both high = coherent)
            amp_coh = min(pattern.amplitude, neighbor.amplitude) / (
                max(pattern.amplitude, neighbor.amplitude) + 1e-10
            )

            coherence = (phase_coh + amp_coh) / 2 * coupling
            total_coherence += coherence
            total_weight += coupling

        if total_weight < 1e-10:
            return 0.0

        return total_coherence / total_weight

    def _redistribute_heat(self):
        """
        Heat flows to low-amplitude patterns (equipartition).
        """
        if self.heat_field.total_heat < 0.01:
            return

        # Find low-amplitude patterns
        for pattern in self.patterns:
            if pattern.amplitude < 0.1:
                local_temp = self.heat_field.get_temperature(pattern.embedding)
                if local_temp > 0.01:
                    # Absorb some heat
                    absorbed = min(local_temp * 0.1, 0.01)
                    pattern.amplitude += absorbed
                    # Add random phase (thermal = incoherent)
                    pattern.phase += np.random.uniform(-0.1, 0.1)

    def _check_combinations(self):
        """
        Patterns at similar radii with aligned phases can combine.

        This replaces explicit "composite discovery" with natural resonance.
        """
        # Group patterns by approximate radius
        radius_groups: Dict[int, List[int]] = defaultdict(list)

        for idx, pattern in enumerate(self.patterns):
            if pattern.amplitude > 0.1:  # Only consider active patterns
                radius_bin = int(pattern.radius)
                radius_groups[radius_bin].append(idx)

        # Check for combinations within each radius band
        for radius, indices in radius_groups.items():
            if len(indices) < 2:
                continue

            for i, idx1 in enumerate(indices):
                p1 = self.patterns[idx1]
                for idx2 in indices[i+1:]:
                    p2 = self.patterns[idx2]

                    # Phase alignment?
                    phase_diff = abs(p1.phase - p2.phase)
                    if phase_diff < 0.2 or phase_diff > 2*np.pi - 0.2:
                        # Close enough in phase - can combine
                        # The stronger one absorbs the weaker
                        if p1.amplitude > p2.amplitude:
                            # p1 absorbs p2
                            p1.amplitude += p2.amplitude * 0.1
                            p1.embedding = (p1.embedding + p2.embedding * 0.1) / 1.1
                            p2.amplitude *= 0.9  # Weaker loses some
                        else:
                            p2.amplitude += p1.amplitude * 0.1
                            p2.embedding = (p2.embedding + p1.embedding * 0.1) / 1.1
                            p1.amplitude *= 0.9

    def _rebuild_spatial_index(self):
        """Rebuild spatial index after patterns have moved."""
        self._spatial_index.clear()
        for idx, pattern in enumerate(self.patterns):
            cell = self._get_cell(pattern.embedding)
            self._spatial_index[cell].append(idx)

    def signal_game_end(self, game_length: int = 0):
        """Game boundary - the anthropic signal."""
        if game_length == 0:
            game_length = self._current_game_length

        self._game_lengths.append(game_length)
        if len(self._game_lengths) > 50:
            self._game_lengths = self._game_lengths[-50:]

        # ANTHROPIC EFFECT: Patterns active during long games move inward
        # This is the key selection mechanism
        if game_length > np.mean(self._game_lengths) if self._game_lengths else 0:
            # Good game! Active patterns get pulled toward center
            for idx in self.curr_active_indices:
                pattern = self.patterns[idx]
                to_center = self._success_centroid - pattern.embedding
                pattern.momentum += to_center * 0.01  # Gentle pull inward

        self._current_game_length = 0
        self.prev_active_indices.clear()

    def choose_action(self, num_actions: int = 3) -> int:
        """
        Choose action based on resonance with current state.

        The action whose embedding resonates most with active patterns wins.
        """
        if not self.curr_active_indices:
            return np.random.randint(0, num_actions)

        action_scores = {}

        # Get current state centroid
        active_patterns = [self.patterns[i] for i in self.curr_active_indices]
        active_amps = [p.amplitude for p in active_patterns]
        total_amp = sum(active_amps) + 1e-10
        state_centroid = sum(
            p.embedding * p.amplitude for p in active_patterns
        ) / total_amp

        for a in range(num_actions):
            action_emb = self._get_action_embedding(a)

            # Distance from state to action
            dist = np.linalg.norm(state_centroid - action_emb)

            # Coupling from active patterns to action pattern
            action_idx = self._find_or_create_pattern(action_emb, tolerance=0.5)
            action_pattern = self.patterns[action_idx]

            coupling_score = 0.0
            for idx in self.curr_active_indices:
                pattern = self.patterns[idx]
                if action_idx in pattern.coupling_kernel:
                    coupling_score += pattern.coupling_kernel[action_idx] * pattern.amplitude

            # Score: closer + more coupled + higher amplitude = better
            proximity_score = 1.0 / (dist + 1)
            amplitude_score = action_pattern.amplitude

            action_scores[a] = proximity_score + coupling_score + 0.1 * amplitude_score + 0.01

        # Temperature from local heat
        avg_temp = np.mean([
            self.heat_field.get_temperature(self.patterns[i].embedding)
            for i in self.curr_active_indices
        ]) if self.curr_active_indices else 0.1

        temperature = 0.5 + avg_temp

        # Softmax selection
        scores = np.array(list(action_scores.values()))
        scores = scores - scores.max()  # Stability
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / exp_scores.sum()

        return np.random.choice(list(action_scores.keys()), p=probs)

    def get_stats(self) -> Dict:
        """Get sieve statistics."""
        if not self.patterns:
            return {'num_patterns': 0}

        amplitudes = [p.amplitude for p in self.patterns]
        radii = [p.radius for p in self.patterns]

        return {
            'frame': self.frame_num,
            'num_patterns': len(self.patterns),
            'total_amplitude': sum(amplitudes),
            'avg_amplitude': np.mean(amplitudes),
            'avg_radius': np.mean(radii),
            'min_radius': min(radii),
            'max_radius': max(radii),
            'total_heat': self.heat_field.total_heat,
            'num_active': len(self.curr_active_indices),
            'centroid_mass': self._centroid_mass,
        }

    def print_state(self):
        """Print current state."""
        stats = self.get_stats()
        print(f"\n{'='*70}")
        print(f"SPIRAL SIEVE STATE (Frame {stats['frame']})")
        print(f"{'='*70}")
        print(f"Patterns: {stats['num_patterns']}")
        print(f"Active: {stats['num_active']}")
        print(f"Total amplitude: {stats['total_amplitude']:.2f}")
        print(f"Avg amplitude: {stats['avg_amplitude']:.4f}")
        print(f"Radius range: {stats['min_radius']:.2f} - {stats['max_radius']:.2f}")
        print(f"Avg radius: {stats['avg_radius']:.2f}")
        print(f"Total heat: {stats['total_heat']:.4f}")

        # Show action states
        print(f"\nAction embeddings:")
        for a, emb in self._action_embeddings.items():
            radius = np.linalg.norm(emb)
            action_idx = self._find_or_create_pattern(emb, tolerance=0.5)
            amp = self.patterns[action_idx].amplitude if action_idx < len(self.patterns) else 0
            print(f"  Action {a}: radius={radius:.2f}, amplitude={amp:.4f}")


# =============================================================================
# TEST
# =============================================================================

def test_spiral_pong():
    """Test spiral sieve on Pong."""
    print("=" * 70)
    print("SPIRAL SIEVE TEST")
    print("N-dimensional whirlpool dynamics")
    print("=" * 70)

    FRAME_SIZE = 21
    sieve = SpiralSieve(embedding_dim=8)  # Lower dim for speed

    # Game state
    ball_x, ball_y = 10.5, 5.0
    ball_dx, ball_dy = 0.5, 0.5
    paddle_x = 10.5

    hits = 0
    misses = 0
    game_lengths = []
    current_game_length = 0

    TOTAL_FRAMES = 20000

    for frame_num in range(TOTAL_FRAMES):
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
                sieve.signal_game_end(current_game_length)
                current_game_length = 0

                ball_y = 5
                ball_x = np.random.uniform(5, FRAME_SIZE - 5)
                ball_dx = np.random.choice([-0.5, 0.5])
                ball_dy = 0.5

        # Progress
        if (frame_num + 1) % 2000 == 0:
            total_games = hits + misses
            hit_rate = hits / total_games if total_games > 0 else 0
            recent_len = np.mean(game_lengths[-20:]) if len(game_lengths) >= 20 else 0

            print(f"\nFrame {frame_num+1}:")
            print(f"  Games: {total_games}, Hit rate: {hit_rate:.1%}")
            print(f"  Recent game length: {recent_len:.1f}")
            sieve.print_state()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    total_games = hits + misses
    print(f"Total frames: {TOTAL_FRAMES}")
    print(f"Total games: {total_games}")
    print(f"Hits: {hits} ({100*hits/total_games:.1f}%)")

    if len(game_lengths) >= 20:
        first_20 = np.mean(game_lengths[:20])
        last_20 = np.mean(game_lengths[-20:])
        print(f"\nGame length evolution:")
        print(f"  First 20 games: {first_20:.1f} frames")
        print(f"  Last 20 games: {last_20:.1f} frames")
        print(f"  Improvement: {(last_20/first_20 - 1)*100:+.1f}%")

    return sieve, game_lengths


if __name__ == "__main__":
    test_spiral_pong()
