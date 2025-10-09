# Global State Implementation Guide for Motion Clip Weighting

## Overview

This document explains how to implement global state variables for tracking trajectory success and weighting motion clips based on reward returns in the ksim PPO/AMP framework.

## Architecture Summary

### Key Components

1. **Command System** (`ksim/commands.py`)
   - Base class for generating commands/tasks for the agent
   - Commands are per-environment state that gets updated each step

2. **PPO Task** (`ksim/task/ppo.py`)
   - Main training loop for PPO algorithm
   - Manages environment rollouts and model updates
   - Uses `RolloutSharedState` for cross-environment data

3. **AMP Task** (`ksim/task/amp.py`)
   - Extends PPO with adversarial motion priors
   - Already uses global state to store motion clips
   - Example at lines 309-325 showing how to add data to `aux_values`

### State Hierarchy

The framework has three levels of state:

1. **RolloutEnvState** (per-environment) - Line 112-122 in `ksim/task/rl.py`
   - Commands, physics state, model carry, etc.
   - Each environment has its own copy

2. **RolloutSharedState** (global to all environments) - Line 127-133 in `ksim/task/rl.py`
   ```python
   @dataclass(frozen=True)
   class RolloutSharedState:
       physics_model: PhysicsModel
       model_arrs: tuple[PyTree, ...]
       aux_values: xax.FrozenDict[str, PyTree]  # ← USE THIS FOR GLOBAL STATE
       rng: PRNGKeyArray
   ```

3. **RolloutConstants** (static throughout training) - Line 138-150 in `ksim/task/rl.py`
   - Model statics, observations, rewards, etc.
   - Cannot be modified during training

## Solution: Using `aux_values` in RolloutSharedState

The `aux_values` field in `RolloutSharedState` is specifically designed for global state that needs to be shared across all environments.

### Example: AMP Task Already Uses This Pattern

In `ksim/task/amp.py` (lines 302-325):

```python
def _get_shared_state(
    self,
    *,
    rng: PRNGKeyArray,
    mj_model: mujoco.MjModel,
    physics_model: PhysicsModel,
    model_arrs: tuple[PyTree, ...],
) -> RolloutSharedState:
    shared_state = super()._get_shared_state(
        rng=rng,
        mj_model=mj_model,
        physics_model=physics_model,
        model_arrs=model_arrs,
    )
    # Add real motions to global state
    shared_state = replace(
        shared_state,
        aux_values=xax.FrozenDict(
            shared_state.aux_values.unfreeze()
            | {
                REAL_MOTIONS_KEY: self.get_real_motions(mj_model),
            }
        ),
    )
    return shared_state
```

## Implementation Plan for Motion Clip Weighting

### Step 1: Define Global State Structure

Create a dataclass to track motion clip statistics:

```python
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotionClipWeightingState:
    """Global state for tracking motion clip success rates."""
    
    # Number of motion clips
    num_clips: int
    
    # Per-clip statistics (all arrays of shape [num_clips])
    clip_success_counts: Array  # Number of successful trajectories per clip
    clip_total_counts: Array    # Total number of trajectories per clip
    clip_total_rewards: Array   # Sum of rewards for each clip
    clip_sampling_weights: Array  # Current sampling probabilities
    
    # Hyperparameters
    min_weight: float = 0.01    # Minimum sampling weight
    smoothing_factor: float = 0.9  # EMA smoothing for weights
    
    def update_clip_stats(
        self, 
        clip_indices: Array,  # [batch_size]
        successes: Array,     # [batch_size]
        returns: Array,       # [batch_size]
    ) -> "MotionClipWeightingState":
        """Update statistics after a batch of trajectories."""
        
        # Count successes and totals per clip
        def update_counts(clip_idx):
            mask = clip_indices == clip_idx
            success_count = jnp.sum(jnp.where(mask, successes, 0))
            total_count = jnp.sum(mask)
            total_reward = jnp.sum(jnp.where(mask, returns, 0.0))
            return success_count, total_count, total_reward
        
        updates = jax.vmap(update_counts)(jnp.arange(self.num_clips))
        new_success_counts, new_total_counts, new_total_rewards = updates
        
        # Update cumulative counts
        success_counts = self.clip_success_counts + new_success_counts
        total_counts = self.clip_total_counts + new_total_counts
        total_rewards = self.clip_total_rewards + new_total_rewards
        
        # Compute success rates
        success_rates = jnp.where(
            total_counts > 0,
            success_counts / total_counts,
            0.5  # Default for clips without data
        )
        
        # Compute average rewards
        avg_rewards = jnp.where(
            total_counts > 0,
            total_rewards / total_counts,
            0.0
        )
        
        # Invert success rate to weight harder clips more
        # Higher weight = lower success rate = harder clip
        inverse_success = 1.0 - success_rates
        
        # Could also use reward-based weighting
        # inverse_reward = 1.0 / (avg_rewards + 1e-6)
        
        # Normalize to get new weights
        raw_weights = inverse_success + self.min_weight
        new_weights = raw_weights / jnp.sum(raw_weights)
        
        # EMA smoothing with previous weights
        smoothed_weights = (
            self.smoothing_factor * self.clip_sampling_weights +
            (1 - self.smoothing_factor) * new_weights
        )
        # Renormalize after smoothing
        smoothed_weights = smoothed_weights / jnp.sum(smoothed_weights)
        
        return MotionClipWeightingState(
            num_clips=self.num_clips,
            clip_success_counts=success_counts,
            clip_total_counts=total_counts,
            clip_total_rewards=total_rewards,
            clip_sampling_weights=smoothed_weights,
            min_weight=self.min_weight,
            smoothing_factor=self.smoothing_factor,
        )
```

### Step 2: Initialize Global State

Override `_get_shared_state` in your task:

```python
class YourAMPTask(AMPTask[Config]):
    
    def _get_shared_state(
        self,
        *,
        rng: PRNGKeyArray,
        mj_model: mujoco.MjModel,
        physics_model: PhysicsModel,
        model_arrs: tuple[PyTree, ...],
    ) -> RolloutSharedState:
        # Get base shared state (includes real motions from AMPTask)
        shared_state = super()._get_shared_state(
            rng=rng,
            mj_model=mj_model,
            physics_model=physics_model,
            model_arrs=model_arrs,
        )
        
        # Get number of motion clips
        real_motions = shared_state.aux_values[REAL_MOTIONS_KEY]
        num_clips = jax.tree_util.tree_leaves(real_motions)[0].shape[0]
        
        # Initialize motion clip weighting state
        weighting_state = MotionClipWeightingState(
            num_clips=num_clips,
            clip_success_counts=jnp.zeros(num_clips),
            clip_total_counts=jnp.zeros(num_clips),
            clip_total_rewards=jnp.zeros(num_clips),
            clip_sampling_weights=jnp.ones(num_clips) / num_clips,  # Uniform initially
        )
        
        # Add to shared state
        shared_state = replace(
            shared_state,
            aux_values=xax.FrozenDict(
                shared_state.aux_values.unfreeze()
                | {
                    "motion_clip_weighting": weighting_state,
                }
            ),
        )
        
        return shared_state
```

### Step 3: Track Which Clip Each Environment Uses

Modify your reset logic to track which motion clip was used. This can be stored in the environment state or in the trajectory's `aux_outputs`:

```python
class YourAMPTask(AMPTask[Config]):
    
    def postprocess_trajectory(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
        trajectory: Trajectory,
        rng: PRNGKeyArray,
    ) -> Trajectory:
        trajectory = super().postprocess_trajectory(
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
            trajectory=trajectory,
            rng=rng,
        )
        
        # Add clip index to trajectory if you're tracking it
        # This assumes you've stored it somewhere in env_states or can infer it
        aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
        # aux_outputs["motion_clip_index"] = ...  # Get from wherever you stored it
        
        return replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))
```

### Step 4: Update Global State After Each Rollout

Override the `update_model` method to update statistics:

```python
class YourAMPTask(AMPTask[Config]):
    
    def update_model(
        self,
        *,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array]]:
        # Get current weighting state
        weighting_state = carry.shared_state.aux_values["motion_clip_weighting"]
        
        # Extract trajectory data
        clip_indices = trajectories.aux_outputs["motion_clip_index"]  # [num_envs]
        successes = trajectories.success[..., -1]  # Final success flag [num_envs]
        
        # Compute returns for each trajectory
        returns = rewards.total.sum(axis=-1)  # Sum over time dimension [num_envs]
        
        # Update statistics
        new_weighting_state = weighting_state.update_clip_stats(
            clip_indices=clip_indices,
            successes=successes,
            returns=returns,
        )
        
        # Update carry with new statistics
        carry = replace(
            carry,
            shared_state=replace(
                carry.shared_state,
                aux_values=xax.FrozenDict(
                    carry.shared_state.aux_values.unfreeze()
                    | {"motion_clip_weighting": new_weighting_state}
                ),
            ),
        )
        
        # Continue with normal PPO/AMP update
        carry, metrics = super().update_model(
            constants=constants,
            carry=carry,
            trajectories=trajectories,
            rewards=rewards,
            rng=rng,
        )
        
        # Add weighting metrics to logging
        metrics = xax.FrozenDict(
            metrics.unfreeze()
            | {
                "motion_clip_weights_min": new_weighting_state.clip_sampling_weights.min(),
                "motion_clip_weights_max": new_weighting_state.clip_sampling_weights.max(),
                "motion_clip_weights_std": new_weighting_state.clip_sampling_weights.std(),
            }
        )
        
        return carry, metrics
```

### Step 5: Use Weights for Motion Clip Sampling

Modify your reset logic to use the learned weights:

```python
class WeightedMotionStateReset(Reset):
    """Resets using weighted sampling of motion clips."""
    
    def __call__(
        self, 
        data: PhysicsData, 
        curriculum_level: Array, 
        rng: PRNGKeyArray,
        weighting_state: MotionClipWeightingState,  # Pass from shared state
        reference_motions: PyTree,  # Pass from shared state
    ) -> tuple[PhysicsData, int]:
        # Sample clip index using learned weights
        clip_index = jax.random.choice(
            rng,
            weighting_state.num_clips,
            p=weighting_state.clip_sampling_weights
        )
        
        # Get the selected clip
        selected_clip = jax.tree_map(lambda x: x[clip_index], reference_motions)
        
        # Sample a frame from the clip
        num_frames = jax.tree_util.tree_leaves(selected_clip)[0].shape[0]
        frame_index = jax.random.randint(rng, (1,), 0, num_frames)[0]
        
        # Set qpos/qvel from the selected frame
        # ... (similar to InitialMotionStateReset)
        
        return data, clip_index  # Return clip_index to track it
```

## Key Considerations

### JAX Constraints

Since all state must be JAX-compatible:
- Use `jax.numpy` arrays, not Python lists or dicts
- All operations must be pure functions
- State updates return new state objects (immutable)
- Use `jax.tree_util.register_dataclass` for custom state classes

### Performance

- The global state is updated once per batch (not per step)
- Statistics are computed efficiently using JAX vectorization
- EMA smoothing prevents rapid weight changes

### Alternative Approaches

1. **Per-Environment Tracking**: Store clip index in `RolloutEnvState.aux_values` or trajectory metadata
2. **Command-Based**: Create a custom `Command` that tracks and updates weights
3. **Separate Update Loop**: Update weights asynchronously in a separate process (more complex)

## Summary

**Where to store global state:** `RolloutSharedState.aux_values`

**When to update:** In `update_model()` after collecting trajectories

**How to use:** Pass values from `shared_state.aux_values` to your reset/sampling functions

The framework already has the infrastructure via the `aux_values` field - you just need to:
1. Initialize your global state in `_get_shared_state()`
2. Update it in `update_model()` or `postprocess_trajectory()`
3. Access it via `shared_state.aux_values[your_key]`

This follows the same pattern already used by AMPTask for storing motion clips globally.
