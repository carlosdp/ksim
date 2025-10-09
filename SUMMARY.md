# Summary: Global State for Motion Clip Weighting

## Key Findings

### 1. **Where to Store Global State**

The framework provides `RolloutSharedState.aux_values` specifically for global state that needs to be shared across all environments.

**Location:** `ksim/task/rl.py`, lines 127-133

```python
@dataclass(frozen=True)
class RolloutSharedState:
    physics_model: PhysicsModel
    model_arrs: tuple[PyTree, ...]
    aux_values: xax.FrozenDict[str, PyTree]  # ← Store global state here
    rng: PRNGKeyArray
```

### 2. **Existing Pattern: AMPTask**

The `AMPTask` already uses this pattern to store motion clips globally (see `ksim/task/amp.py`, lines 302-325):

```python
def _get_shared_state(...) -> RolloutSharedState:
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

### 3. **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         RolloutSharedState (GLOBAL)                │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │ aux_values:                              │     │    │
│  │  │   - real_motions (from AMPTask)         │     │    │
│  │  │   - motion_clip_weighting ← NEW         │     │    │
│  │  │   - ... (any other global state)        │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           │ Shared across all envs          │
│                           ↓                                  │
│  ┌──────────────────────────────────────────────────┐      │
│  │  RolloutEnvState[0]  │  [1]  │  [2]  │  ...      │      │
│  │  (per-environment)                                │      │
│  │  - commands                                       │      │
│  │  - physics_state                                  │      │
│  │  - reward_carry                                   │      │
│  └──────────────────────────────────────────────────┘      │
│                                                              │
│  Rollout → Postprocess → Update Model → Repeat              │
└─────────────────────────────────────────────────────────────┘
```

### 4. **Implementation Steps**

#### Step 1: Define State Structure
Create a JAX-compatible dataclass with `@jax.tree_util.register_dataclass`:

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotionClipWeightingState:
    num_clips: int
    clip_success_counts: Array  # [num_clips]
    clip_total_counts: Array    # [num_clips]
    sampling_weights: Array     # [num_clips]
    # ... more fields
```

#### Step 2: Initialize in `_get_shared_state()`
Override this method to add your global state:

```python
def _get_shared_state(self, ...) -> RolloutSharedState:
    shared_state = super()._get_shared_state(...)
    
    weighting_state = MotionClipWeightingState.create(num_clips=...)
    
    shared_state = replace(
        shared_state,
        aux_values=xax.FrozenDict(
            shared_state.aux_values.unfreeze()
            | {"motion_clip_weighting": weighting_state}
        ),
    )
    return shared_state
```

#### Step 3: Track Clip Usage
You need to track which clip each environment uses. Options:

1. **Store in trajectory aux_outputs** (recommended):
   ```python
   def postprocess_trajectory(...) -> Trajectory:
       aux_outputs = {..., "motion_clip_index": clip_index}
       trajectory = replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))
   ```

2. **Store in RolloutEnvState**:
   Add a field to track current clip (more complex, requires modifying state structure)

3. **Store in Command**:
   Create a custom command that tracks the clip

#### Step 4: Update Global State
Update statistics in `update_model()`:

```python
def update_model(self, ...) -> tuple[RLLoopCarry, xax.FrozenDict]:
    weighting_state = carry.shared_state.aux_values["motion_clip_weighting"]
    
    # Extract data from trajectories
    clip_indices = trajectories.aux_outputs["motion_clip_index"]
    successes = trajectories.success[..., -1]
    returns = rewards.total.sum(axis=-1)
    
    # Update statistics
    new_state = weighting_state.update(clip_indices, successes, returns)
    
    # Update carry
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
    
    # Continue with normal update
    return super().update_model(...)
```

#### Step 5: Use Weights for Sampling
Access weights when resetting environments:

```python
def reset_with_weighted_sampling(data, rng, shared_state):
    weighting_state = shared_state.aux_values["motion_clip_weighting"]
    
    # Sample clip using learned weights
    clip_idx = jax.random.choice(
        rng,
        weighting_state.num_clips,
        p=weighting_state.sampling_weights
    )
    
    # Use selected clip...
```

### 5. **Key Files to Review**

1. **`ksim/task/rl.py`**: 
   - Lines 127-133: `RolloutSharedState` definition
   - Lines 2189-2198: `_get_shared_state()` default implementation

2. **`ksim/task/amp.py`**:
   - Lines 302-325: Example of adding to `aux_values`
   - Lines 570-625: Motion clip sampling in `_make_real_batch()`

3. **`ksim/commands.py`**:
   - Shows how to create custom command types
   - Could be used to track/manage clip selection

4. **`ksim/resets.py`**:
   - Lines 292-313: `InitialMotionStateReset` - samples from motion clips
   - Can be extended to use weighted sampling

### 6. **Important Constraints**

**JAX Compatibility:**
- All state must be JAX arrays or PyTrees
- Functions must be pure (no side effects)
- State is immutable (use `replace()` to update)
- Use `jax.tree_util.register_dataclass` for custom types

**Performance:**
- Update frequency: Once per batch (not per step)
- Use vectorized operations (`jax.vmap`)
- Avoid Python loops over environments

**Shared State Updates:**
- Only update in `update_model()` or similar batch-level functions
- Don't update per environment or per step
- Changes apply globally to all environments

### 7. **Code Examples Provided**

1. **`GLOBAL_STATE_IMPLEMENTATION_GUIDE.md`**: 
   - Comprehensive guide with architecture explanation
   - Step-by-step implementation details
   - Alternative approaches

2. **`example_motion_clip_weighting.py`**:
   - Complete working example
   - `MotionClipWeightingState` dataclass
   - `MotionClipWeightedAMPTask` implementation
   - `WeightedInitialMotionStateReset` for sampling
   - Logging and metrics

### 8. **Next Steps**

To implement this in your codebase:

1. **Copy the pattern** from `example_motion_clip_weighting.py`
2. **Modify your reset logic** to track which clip is used
3. **Store clip index** in trajectory `aux_outputs`
4. **Override `_get_shared_state()`** to initialize weighting state
5. **Override `update_model()`** to update statistics
6. **Test** with logging to verify weights are updating correctly

### 9. **Alternative: Simpler Approaches**

If full weighting is too complex initially, consider:

1. **Logging Only**: Just track statistics without updating weights
2. **Manual Bins**: Manually define "easy" vs "hard" clip bins
3. **Curriculum-Based**: Use existing curriculum system to gradually introduce clips

### 10. **Questions to Resolve**

1. **How are clips currently selected?** 
   - Check your reset logic to see if you're using `InitialMotionStateReset`
   - Determine where clip selection happens

2. **Where to store clip index?**
   - Trajectory aux_outputs (easiest)
   - Environment state (more complex)
   - Command (alternative)

3. **What weighting strategy?**
   - Inverse success rate (harder clips weighted more)
   - Inverse reward (lower reward clips weighted more)
   - Combination of both
   - Other custom metric

## Conclusion

The framework already has infrastructure for global state via `RolloutSharedState.aux_values`. You just need to:

1. ✅ Define your state structure (JAX-compatible dataclass)
2. ✅ Initialize it in `_get_shared_state()`
3. ✅ Track clip usage per environment
4. ✅ Update statistics in `update_model()`
5. ✅ Use weights when sampling clips

The provided example code gives you a complete implementation to adapt to your specific use case.
