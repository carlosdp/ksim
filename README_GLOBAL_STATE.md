# Global State Implementation for Motion Clip Weighting

This directory contains documentation and examples for implementing global state variables to track trajectory success and weight motion clips in the ksim PPO/AMP framework.

## 📁 Files

### Documentation
- **`SUMMARY.md`** - Quick reference guide with key findings and implementation steps
- **`GLOBAL_STATE_IMPLEMENTATION_GUIDE.md`** - Comprehensive architectural guide

### Code Examples
- **`example_motion_clip_weighting.py`** - Complete implementation of motion clip weighting system
- **`clip_tracking_example.py`** - Multiple approaches for tracking which clip each environment uses

## 🎯 Quick Start

### Understanding the Architecture

The ksim framework has three levels of state:

1. **RolloutEnvState** - Per-environment (each env has its own)
2. **RolloutSharedState** - Global across all envs (what we need!)
3. **RolloutConstants** - Static configuration (unchanging)

For global state, use `RolloutSharedState.aux_values`:

```python
@dataclass(frozen=True)
class RolloutSharedState:
    physics_model: PhysicsModel
    model_arrs: tuple[PyTree, ...]
    aux_values: xax.FrozenDict[str, PyTree]  # ← Store global state here!
    rng: PRNGKeyArray
```

### Implementation Checklist

- [ ] **Define state structure** - Create JAX-compatible dataclass
- [ ] **Initialize in `_get_shared_state()`** - Add state to `aux_values`
- [ ] **Track clip indices** - Store which clip each env uses
- [ ] **Update in `update_model()`** - Update statistics after rollouts
- [ ] **Use for sampling** - Sample clips with learned weights

### Minimal Example

```python
from dataclasses import dataclass, replace
import jax.numpy as jnp
import xax

# 1. Define global state
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotionClipWeightingState:
    num_clips: int
    sampling_weights: Array  # [num_clips]
    # ... other fields

# 2. Initialize in task
class MyTask(AMPTask):
    def _get_shared_state(self, ...) -> RolloutSharedState:
        shared_state = super()._get_shared_state(...)
        
        # Add weighting state
        weighting_state = MotionClipWeightingState(
            num_clips=10,
            sampling_weights=jnp.ones(10) / 10,
        )
        
        shared_state = replace(
            shared_state,
            aux_values=xax.FrozenDict(
                shared_state.aux_values.unfreeze()
                | {"motion_clip_weighting": weighting_state}
            ),
        )
        return shared_state
    
    # 3. Update in update_model
    def update_model(self, ...) -> ...:
        weighting_state = carry.shared_state.aux_values["motion_clip_weighting"]
        
        # Update based on trajectory success
        new_state = weighting_state.update(...)
        
        # Store back in carry
        carry = replace(
            carry,
            shared_state=replace(
                carry.shared_state,
                aux_values=xax.FrozenDict(
                    carry.shared_state.aux_values.unfreeze()
                    | {"motion_clip_weighting": new_state}
                ),
            ),
        )
        return carry, metrics
```

## 🔍 Key Insights

### Where AMPTask Already Does This

Look at `ksim/task/amp.py` lines 302-325:

```python
def _get_shared_state(self, ...) -> RolloutSharedState:
    shared_state = super()._get_shared_state(...)
    shared_state = replace(
        shared_state,
        aux_values=xax.FrozenDict(
            shared_state.aux_values.unfreeze()
            | {REAL_MOTIONS_KEY: self.get_real_motions(mj_model)}
        ),
    )
    return shared_state
```

This is the **exact pattern** to follow for adding global state!

### Common Pitfall: Tracking Clip Indices

The trickiest part is tracking which clip each environment uses. See `clip_tracking_example.py` for five different approaches:

1. ✅ **Use Command system** (recommended) - Clean and integrated
2. Store in trajectory aux_outputs - Simple but manual
3. Store in environment state - Requires framework modification
4. Infer from reset data - Hacky but works
5. Use event system - Complex but flexible

## 📊 What Gets Tracked

The weighting system tracks per-clip:
- **Success counts** - How many trajectories succeeded
- **Failure counts** - How many trajectories failed
- **Total rewards** - Sum of episode returns
- **Sampling weights** - Learned probabilities (updated via EMA)

## 🎓 Weighting Strategies

### Inverse Success Rate (Default)
Weight harder clips (lower success rate) more heavily:
```python
inverse_success = 1.0 - success_rate
weights = inverse_success / sum(inverse_success)
```

### Inverse Reward
Weight clips with lower rewards:
```python
inverse_reward = 1.0 - normalized_reward
weights = inverse_reward / sum(inverse_reward)
```

### Combined
Use both metrics:
```python
weights = (inverse_success + inverse_reward) / 2
```

## 🔧 JAX Requirements

All state must be JAX-compatible:
- ✅ Use `jnp.array` not Python lists
- ✅ Use `@jax.tree_util.register_dataclass`
- ✅ Make dataclasses `frozen=True`
- ✅ Pure functions only (no side effects)
- ✅ Immutable updates via `replace()`

## 📈 Logging and Metrics

The example implementation provides metrics:
```python
def get_metrics(self) -> dict[str, Array]:
    return {
        "motion_clip/weights_min": ...,
        "motion_clip/weights_max": ...,
        "motion_clip/success_rate_mean": ...,
        "motion_clip/avg_reward_mean": ...,
    }
```

Add these to your training metrics in `update_model()`.

## 🐛 Debugging Tips

1. **Log initial state** - Verify initialization in `_get_shared_state()`
2. **Check shapes** - All arrays should have consistent batch dimensions
3. **Print weights** - Watch how weights evolve during training
4. **Validate probabilities** - Weights should sum to 1.0
5. **Test with small examples** - Use 2-3 clips first

## 🔗 Related Code

### Framework Files
- `ksim/task/rl.py` - RolloutSharedState definition
- `ksim/task/ppo.py` - PPO task and update logic
- `ksim/task/amp.py` - AMP task (uses global state for motions)
- `ksim/commands.py` - Command system
- `ksim/resets.py` - Reset logic (where clips are sampled)

### Key Methods to Override
- `_get_shared_state()` - Initialize global state
- `update_model()` - Update statistics after rollouts
- `postprocess_trajectory()` - Add clip indices to trajectories
- `get_commands()` - Add motion clip command (optional)

## 💡 Alternative Approaches

If full implementation is too complex:

1. **Logging only** - Track statistics without updating weights
2. **Manual binning** - Define "easy" vs "hard" clips manually
3. **Curriculum-based** - Use existing curriculum to introduce clips
4. **Periodic rebalancing** - Update weights every N iterations, not continuously

## 🚀 Next Steps

1. Read `SUMMARY.md` for quick overview
2. Study `example_motion_clip_weighting.py` for full implementation
3. Choose a clip tracking approach from `clip_tracking_example.py`
4. Implement in your task by:
   - Defining `MotionClipWeightingState`
   - Overriding `_get_shared_state()`
   - Overriding `update_model()`
   - Adding clip tracking (via Command or aux_outputs)
5. Test with logging and small number of clips
6. Scale up once working

## ❓ Questions?

Key questions to answer for your implementation:

1. **How many motion clips?** - Determines state array sizes
2. **What triggers clip switching?** - Episode end? Fixed interval?
3. **Weighting by success or reward?** - Or both?
4. **How to track clip indices?** - Command? aux_outputs? Other?
5. **Update frequency?** - Every batch? Every N batches?

## 📚 References

- **JAX documentation**: https://jax.readthedocs.io/
- **PPO paper**: Schulman et al. 2017
- **AMP paper**: Adversarial Motion Priors

## ✨ Summary

The framework provides `RolloutSharedState.aux_values` for exactly this use case. Follow the pattern already used by `AMPTask` for storing motion clips, and extend it to track success statistics and sampling weights. The provided examples give you a complete working implementation to adapt to your needs.
