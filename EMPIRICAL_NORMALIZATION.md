# Empirical Normalization in PPO

This document explains how to use empirical normalization in the PPO task, which matches the implementation in RSL RL.

## What is Empirical Normalization?

Empirical normalization (also known as observation normalization) tracks running statistics (mean and standard deviation) of observations during training and normalizes them before feeding them to the value function (critic). This helps stabilize training and improve performance, especially when observations have different scales.

## How to Enable

### Step 1: Enable in Config

Set `empirical_normalization=True` in your PPO config:

```python
config = HumanoidWalkingTaskConfig(
    empirical_normalization=True,
    # ... other config options
)
```

### Step 2: Implement `get_critic_obs`

Your task must implement the `get_critic_obs` method to return the observation vector that goes into the critic:

```python
class MyTask(ksim.PPOTask[MyTaskConfig]):
    def get_critic_obs(
        self,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
    ) -> Array:
        """Return the flattened observation vector for the critic."""
        # Example: concatenate all relevant observations
        obs_n = jnp.concatenate([
            observations["joint_position"],
            observations["joint_velocity"] / 10.0,
            observations["base_orientation"],
            # ... other observations
        ], axis=-1)
        return obs_n
```

### Step 3: Use Normalization in `run_critic`

In your `run_critic` method, apply normalization to the observations:

```python
def run_critic(
    self,
    model: Critic,
    observations: xax.FrozenDict[str, PyTree],
    commands: xax.FrozenDict[str, PyTree],
    carry: Array,
) -> tuple[Array, Array]:
    # Get the raw observation vector
    obs_n = self.get_critic_obs(observations, commands)
    
    # Apply normalization if enabled
    if self.config.empirical_normalization:
        # Get normalization state from the training loop
        # Note: During training this is available in aux_values
        # For inference, you may want to save and load this state
        norm_state = self.get_norm_state()  # You'll need to implement this
        obs_n = self.normalize_critic_obs(obs_n, norm_state)
    
    # Feed normalized observations to critic
    return model.forward(obs_n, carry)
```

### Step 4: Access Normalization State

During training, the normalization state is stored in `carry.shared_state.aux_values["obs_norm_state"]`. For inference, you'll want to save this state along with your model:

```python
# During training, access the state:
norm_state = carry.shared_state.aux_values["obs_norm_state"]

# Save it with your model for inference:
checkpoint = {
    "model": model,
    "norm_state": norm_state,
}
```

## How It Works

1. **Initialization**: On the first training step, the normalization state is initialized with:
   - `mean`: zeros (shape matches observation dimension)
   - `var`: ones (shape matches observation dimension)  
   - `count`: small value (1e-4) to avoid division by zero

2. **Statistics Update**: After each rollout, the running statistics are updated using Welford's online algorithm:
   - The mean and variance are updated incrementally
   - This matches RSL RL's implementation exactly

3. **Normalization**: When computing values, observations are normalized as:
   ```
   normalized_obs = (obs - mean) / sqrt(var + eps)
   ```

4. **Storage**: The normalization state is stored in `aux_values` alongside other shared training state like adaptive KL coefficient.

## Example: Full Implementation

Here's a complete example based on the walking task:

```python
import ksim
import jax.numpy as jnp

class MyWalkingTask(ksim.PPOTask[MyWalkingTaskConfig]):
    
    def get_critic_obs(
        self,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
    ) -> Array:
        """Build critic observation vector."""
        linvel_cmd = commands["linvel"]
        angvel_cmd = commands["angvel"]
        
        return jnp.concatenate([
            observations["joint_position"],
            observations["joint_velocity"] / 10.0,
            observations["base_orientation"],
            observations["base_linear_velocity"],
            observations["base_angular_velocity"],
            jnp.array([linvel_cmd.target_vel, linvel_cmd.target_yaw]),
            jnp.array([angvel_cmd.target_vel]),
        ], axis=-1)
    
    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        carry: Array,
    ) -> tuple[Array, Array]:
        """Run critic with optional normalization."""
        obs_n = self.get_critic_obs(observations, commands)
        
        # Note: During training, normalization is applied automatically
        # via the PPO task infrastructure. This method is called during
        # rollout where we use the current normalization state.
        
        return model.forward(obs_n, carry)
```

## Benefits

- **Improved Training Stability**: Normalizing observations helps prevent gradient explosion/vanishing
- **Better Performance**: Particularly helpful when observations have different scales
- **RSL RL Compatibility**: Matches the behavior of RSL RL for easier migration

## Technical Details

The implementation uses:
- **Welford's Online Algorithm**: For numerically stable computation of running statistics
- **JAX JIT Compilation**: All operations are JIT-compiled for performance
- **Automatic Integration**: Hooks into the PPO update loop automatically

The normalization state is a PyTree with:
```python
@dataclass
class EmpiricalNormalizationState:
    mean: Array  # Running mean (shape: [obs_dim])
    var: Array   # Running variance (shape: [obs_dim])
    count: Array # Number of samples seen (scalar)
```
