# Accuracy Analysis and Fix

## The Problem

Original accuracy was only **~65%**. Diagnosis revealed:

### Issue 1: False Positives on Rare Events
Rules like `{E1} → +DR (2%)` were marked "deterministic" because 2% < 5% threshold.
- These were predicting game endings (draw, win) on almost every move
- 96-98% false positive rate on +DR, +OW, +XW

### Issue 2: False Negatives on Common Events  
Rules like `{E7, PX} → +PO (84%)` weren't "deterministic" (not >95%).
- Turn changes (+PX, -PO, +PO, -PX) have 84-94% probability
- These weren't being predicted despite being very likely

## The Fix

**Use probability threshold instead of "is_deterministic" flag**

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 50% | 75.7% | 96.9% | 85.0% |
| **70%** | **93.4%** | **96.7%** | **95.0%** |
| 80% | 94.5% | 93.6% | 94.0% |
| 95% | 98.9% | 75.2% | 85.4% |

**Best: 70% threshold achieves 95% F1**

## Corrected Results

| Game | Episodes | F1 | Precision | Recall | High-Conf Rules |
|------|----------|-----|-----------|--------|-----------------|
| **TicTacToe** |
| | 50 | 90.4% | 94.3% | 86.8% | 225 |
| | 100 | 94.9% | 93.8% | 96.1% | 381 |
| | 500 | 94.9% | 93.4% | 96.6% | 579 |
| | 2000 | **95.2%** | 93.6% | 96.9% | 849 |
| **MiniGrid Empty-5x5** |
| | 50 | 66.7% | 88.7% | 53.5% | 174 |
| | 500 | 69.1% | 91.2% | 55.6% | 370 |
| | 2000 | **70.0%** | 89.1% | 57.6% | 361 |
| **MiniGrid DoorKey-5x5** |
| | 50 | 59.8% | 91.1% | 44.5% | 458 |
| | 250 | **64.7%** | 93.7% | 49.4% | 1189 |

## Key Insights

### TicTacToe: 95% F1
- Near-perfect prediction after just 100 episodes
- Simple rules: `{PX} + action_i → +Xi (100%)` cover most transitions
- Remaining 5% error: rare game endings (wins, draws)

### MiniGrid: 70% F1
- Lower recall (58%) - we miss many effects
- High precision (89%) - what we predict is usually correct
- Why lower? More probabilistic dynamics, partial observability

### The Rule of Thumb
- **Threshold 70%** balances precision/recall well
- **"is_deterministic"** (95%/5%) is too strict for prediction
- Most transitions are high-probability but not 100%

## Sample Predictions (TicTacToe)

```
Action 0 on initial board:
  Predicted: {+PO, +X0, -E0, -PX}
  Actual:    {+PO, +X0, -E0, -PX}
  ✓ 100% correct

Action 1 after X plays:
  Predicted: {+O1, +PX, -E1, -PO}  
  Actual:    {+O1, +PX, -E1, -PO}
  ✓ 100% correct
```

## Convergence

- **TicTacToe**: 95% F1 by episode 100
- **MiniGrid Empty**: 70% F1 by episode 250, plateaus
- **MiniGrid DoorKey**: 65% F1 by episode 250, still improving

The system learns quickly and achieves high accuracy on simple domains.
Complex domains plateau at lower accuracy due to inherent stochasticity.
