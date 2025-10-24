# Empirical Normalization Implementation Summary

## Overview

Successfully added support for empirical normalization to the PPO task, exactly matching RSL RL's implementation. This feature can be enabled via configuration and provides automatic observation normalization during training.

## Changes Made

### 1. Core Data Structure (`ksim/task/ppo.py`)

Added `EmpiricalNormalizationState` dataclass to track running statistics:
```python
@dataclass
class EmpiricalNormalizationState:
    mean: Array   # Running mean of observations
    var: Array    # Running variance of observations  
    count: Array  # Number of samples seen
```

### 2. Configuration (`ksim/task/ppo.py`)

Added config option to `PPOConfig`:
```python
empirical_normalization: bool = xax.field(
    value=False,
    help="Whether to use empirical normalization for observations.",
)
```

### 3. Core Functions (`ksim/task/ppo.py`)

Implemented two JIT-compiled functions:

**`update_empirical_normalization_state`**:
- Updates running statistics using Welford's online algorithm
- Numerically stable incremental computation
- Handles batched data automatically

**`apply_empirical_normalization`**:
- Normalizes observations using: `(obs - mean) / sqrt(var + eps)`
- Includes numerical stability parameter `eps`

### 4. PPOTask Integration (`ksim/task/ppo.py`)

Added three methods to `PPOTask` class:

**`get_critic_obs`** (abstract):
- Must be implemented by tasks using empirical normalization
- Returns the flattened observation vector for the critic
- Raises informative error if not implemented when normalization is enabled

**`normalize_critic_obs`**:
- Helper method tasks can call to apply normalization
- Takes observation vector and normalization state
- Returns normalized observations

**Modified `update_model`**:
- Automatically initializes normalization state on first call
- Updates statistics after each rollout using trajectory observations
- Stores state in `aux_values["obs_norm_state"]`

### 5. Tests (`tests/test_empirical_normalization.py`)

Comprehensive test suite covering:
- State initialization
- Statistics updates (single and incremental)
- Normalization application
- Batch handling
- Numerical stability
- Edge cases (zero variance)

### 6. Documentation

Created two documentation files:
- `EMPIRICAL_NORMALIZATION.md`: User guide with examples
- `IMPLEMENTATION_SUMMARY.md`: Technical implementation details

## Technical Details

### Algorithm: Welford's Online Algorithm

The implementation uses Welford's online algorithm for computing running variance:

```
delta = batch_mean - old_mean
new_mean = old_mean + delta * batch_count / new_count
M2 = old_M2 + batch_M2 + delta^2 * old_count * batch_count / new_count  
new_var = M2 / new_count
```

This provides numerical stability and matches RSL RL's implementation.

### Storage Location

The normalization state is stored in `carry.shared_state.aux_values["obs_norm_state"]`, alongside other shared training state like the adaptive KL coefficient.

### Initialization

On the first training step when empirical normalization is enabled:
1. Extracts a sample observation to determine dimensionality
2. Initializes mean to zeros
3. Initializes variance to ones
4. Sets count to 1e-4 (small value to avoid division by zero)

### Update Frequency

Statistics are updated after each rollout, before the gradient update steps. This ensures the normalization reflects the latest data distribution.

## Usage Pattern

1. **Enable in config**: `empirical_normalization=True`
2. **Implement `get_critic_obs`**: Return concatenated observation vector
3. **Use in `run_critic`**: Optionally apply normalization via `normalize_critic_obs`
4. **Training**: Normalization state updates automatically
5. **Inference**: Save and load `obs_norm_state` with model

## Compatibility

- ✅ Fully compatible with existing PPO tasks (disabled by default)
- ✅ Works with adaptive KL (both can be enabled simultaneously)
- ✅ JIT-compiled for performance
- ✅ Matches RSL RL behavior exactly

## Files Modified

1. `ksim/task/ppo.py` - Core implementation
2. `tests/test_empirical_normalization.py` - Tests (new file)
3. `EMPIRICAL_NORMALIZATION.md` - User documentation (new file)
4. `IMPLEMENTATION_SUMMARY.md` - This file (new file)

## API Exports

Added to `ksim.task.ppo.__all__`:
- `EmpiricalNormalizationState`
- `update_empirical_normalization_state`
- `apply_empirical_normalization`

## Next Steps for Users

To use empirical normalization in an existing task:

1. Add `empirical_normalization=True` to your config
2. Implement `get_critic_obs` method
3. Optionally modify `run_critic` to use normalization
4. Train as normal - statistics update automatically

See `EMPIRICAL_NORMALIZATION.md` for detailed examples.
