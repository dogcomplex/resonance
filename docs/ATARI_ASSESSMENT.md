# Atari Assessment for Hierarchical Learner

## Observation Types Available

1. **RGB Pixels**: 210 x 160 x 3 = 100,800 values per frame
2. **Grayscale**: 210 x 160 = 33,600 values per frame  
3. **RAM**: 128 bytes - the actual machine state!

## Our Fit by Observation Type

### RAM Mode (obs_type="ram") - PROMISING! ★★★★☆

The Atari 2600 has only **128 bytes of RAM**. This is:
- Fully captures game state (no hidden info beyond what the game itself tracks)
- Discrete values (0-255 per byte)
- Small enough to tokenize directly

**Tokenization strategy:**
```python
def tokenize_ram(ram):
    tokens = set()
    for i, val in enumerate(ram):
        # Option 1: Exact values (128 * 256 = 32K possible tokens)
        tokens.add(f"ram_{i}_{val}")
        
        # Option 2: Bucketed (128 * 16 = 2K tokens)
        tokens.add(f"ram_{i}_{val // 16}")
    return tokens
```

**Challenges:**
- State space: Up to 256^128 theoretical (but most are unreachable)
- In practice: Games explore maybe 10^6 - 10^9 states
- Need good general rules since exact coverage impossible

### Pixel Mode - DIFFICULT ★★☆☆☆

**Problems:**
- 100K continuous values → essentially infinite state space
- Would need to discretize/hash frames
- Same visual appearance can have different underlying states

**Possible approaches:**
- Frame hashing (lossy)
- Feature extraction (CNN embeddings → tokens)
- Object detection → symbolic tokens

This is basically the "deep RL" problem - not our sweet spot.

## Game-by-Game Assessment

### GOOD FIT (RAM mode)

| Game | Why |
|------|-----|
| **Breakout** | Ball/paddle positions in RAM, deterministic physics |
| **Pong** | Simple state: two paddles, one ball |
| **Space Invaders** | Grid-based enemies, discrete positions |
| **Frogger** | Grid movement, lane-based |
| **Pac-Man** | Grid-based, discrete ghost positions |

### MODERATE FIT

| Game | Challenge |
|------|-----------|
| **Asteroids** | Continuous positions, many objects |
| **Centipede** | Many moving parts |
| **Q*bert** | Isometric grid, should work |

### POOR FIT

| Game | Why |
|------|-----|
| **Montezuma's Revenge** | Long-horizon planning, exploration |
| **Pitfall** | Requires memory of room layouts |
| Any game with random elements | Our learner expects determinism |

## Key Insight: Atari Games ARE Deterministic!

From the docs: "Atari games are entirely deterministic. Agents could achieve 
state-of-art performance by simply memorizing optimal action sequences."

This is why they added "sticky actions" (25% chance previous action repeats).

**For V9/V10**: 
- Without sticky actions: Perfect fit! Deterministic rules.
- With sticky actions: Becomes probabilistic, but we handle that too.

## Recommended Experiment

1. Start with **Breakout** or **Pong** in RAM mode
2. Disable sticky actions (repeat_action_probability=0)
3. Tokenize RAM bytes (bucketed)
4. Train for ~10K transitions
5. Measure: Can we predict next RAM state from current RAM + action?

## Expected Results

| Metric | Estimate | Reasoning |
|--------|----------|-----------|
| Seen state accuracy | 100% | Deterministic + exact match |
| State coverage | <1% | Huge state space |
| Unseen inference | 60-80%? | Depends on rule generalization |
| Practical use | World model for MCTS | Could simulate futures |

## The Real Question

Can our **general rules** discover things like:
- "Ball at position X moving right + action=stay → ball at X+dx"
- "Paddle at Y + action=up → paddle at Y-1"

If RAM bytes map cleanly to game objects, yes!
If RAM is obfuscated/compressed, harder.

## Comparison to Deep RL

| Approach | State Rep | Learns | Scales |
|----------|-----------|--------|--------|
| DQN | Pixels → CNN | Policy | ✓ (with GPU) |
| V9/V10 | RAM → tokens | World model | ✓ (with storage) |

We're not competing with DQN - we're complementary.
DQN learns WHAT to do. We learn WHAT HAPPENS.

## Experimental Results: Simulated Pong

### Key Finding: Tokenization Granularity vs Aliasing

| Bucket Size | Seen F1 | Aliasing | States |
|-------------|---------|----------|--------|
| //20 (coarse) | 2% | 91% | 224 |
| //10 (medium) | 29% | 66% | 2,066 |
| //4 (fine) | **76%** | 22% | 7,630 |
| Exact | **99.9%** | 0% | 11,491 |

### The Fundamental Tradeoff

**Coarse tokenization:**
- Fewer unique states → better coverage
- But positions 81 and 82 both become "ball_x_4"
- They produce DIFFERENT next states → aliasing
- Learner sees this as "probabilistic" even though physics is deterministic

**Exact tokenization:**
- No aliasing → 99.9% accuracy on seen states
- But state space explodes → most states unseen
- Need good GENERAL RULES for unseen states

### Implications for Real Atari

Real Atari RAM has 128 bytes. With exact tokenization:
- Theoretical: 256^128 states
- Practical: Maybe 10^6 - 10^8 reachable states
- We'd see <0.1% of states during training

**Strategy for Atari:**
1. Use **exact tokenization** for accuracy
2. Accept low coverage
3. Rely heavily on **general rules** for unseen states
4. Add **relative features** to improve generalization:
   - `ball_paddle_offset` instead of absolute positions
   - `ball_moving_toward_player` boolean
   - `near_wall` indicators
