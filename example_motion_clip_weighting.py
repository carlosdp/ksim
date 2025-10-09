"""
Example implementation of motion clip weighting based on trajectory success.

This file demonstrates how to:
1. Track which motion clips are used in each environment
2. Maintain global statistics about clip success rates
3. Weight motion clip sampling based on performance
"""

from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.resets import Reset
from ksim.task.amp import AMPConfig, AMPTask, REAL_MOTIONS_KEY
from ksim.task.rl import (
    RLLoopCarry,
    RLLoopConstants,
    RolloutConstants,
    RolloutEnvState,
    RolloutSharedState,
)
from ksim.types import PhysicsData, PhysicsModel, RewardState, Trajectory
from ksim.utils.mujoco import update_data_field

# Global state key
MOTION_CLIP_WEIGHTING_KEY = "_motion_clip_weighting"
MOTION_CLIP_INDEX_KEY = "_motion_clip_index"


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotionClipWeightingState:
    """Global state for tracking and weighting motion clips.
    
    This state is shared across all environments and updated after each batch
    of trajectories to adjust sampling probabilities.
    """
    
    num_clips: int
    
    # Per-clip cumulative statistics (shape: [num_clips])
    clip_success_counts: Array  # Number of successful episodes
    clip_failure_counts: Array  # Number of failed episodes
    clip_total_reward: Array    # Sum of episode returns
    clip_episode_counts: Array  # Total number of episodes
    
    # Current sampling weights (shape: [num_clips])
    sampling_weights: Array
    
    # Hyperparameters
    min_weight: float = 0.01        # Minimum sampling probability
    ema_alpha: float = 0.9          # Exponential moving average factor
    reward_scale: float = 1.0       # Scale for reward-based weighting
    use_success_weighting: bool = True   # Weight by inverse success rate
    use_reward_weighting: bool = False   # Weight by inverse reward
    
    @classmethod
    def create(
        cls,
        num_clips: int,
        min_weight: float = 0.01,
        ema_alpha: float = 0.9,
        reward_scale: float = 1.0,
        use_success_weighting: bool = True,
        use_reward_weighting: bool = False,
    ) -> "MotionClipWeightingState":
        """Initialize with uniform weights."""
        return cls(
            num_clips=num_clips,
            clip_success_counts=jnp.zeros(num_clips),
            clip_failure_counts=jnp.zeros(num_clips),
            clip_total_reward=jnp.zeros(num_clips),
            clip_episode_counts=jnp.zeros(num_clips),
            sampling_weights=jnp.ones(num_clips) / num_clips,
            min_weight=min_weight,
            ema_alpha=ema_alpha,
            reward_scale=reward_scale,
            use_success_weighting=use_success_weighting,
            use_reward_weighting=use_reward_weighting,
        )
    
    def update(
        self,
        clip_indices: Array,  # [batch_size] - which clip each env used
        episode_done: Array,  # [batch_size] - whether episode finished
        successes: Array,     # [batch_size] - success flag for completed episodes
        returns: Array,       # [batch_size] - total reward for completed episodes
    ) -> "MotionClipWeightingState":
        """Update statistics and recompute sampling weights.
        
        Args:
            clip_indices: Index of motion clip used by each environment
            episode_done: Whether each environment completed an episode
            successes: Success flag for environments that completed
            returns: Total return for environments that completed
            
        Returns:
            New weighting state with updated statistics
        """
        
        # Only count statistics from completed episodes
        def accumulate_clip_stats(clip_idx):
            """Accumulate statistics for a specific clip."""
            # Mask for this clip
            is_this_clip = clip_indices == clip_idx
            is_done_this_clip = is_this_clip & episode_done
            
            # Count episodes
            num_episodes = jnp.sum(is_done_this_clip)
            
            # Count successes/failures
            num_successes = jnp.sum(jnp.where(is_done_this_clip, successes, 0))
            num_failures = num_episodes - num_successes
            
            # Sum rewards (only from completed episodes)
            total_reward = jnp.sum(jnp.where(is_done_this_clip, returns, 0.0))
            
            return num_successes, num_failures, total_reward, num_episodes
        
        # Vectorized accumulation over all clips
        clip_range = jnp.arange(self.num_clips)
        new_stats = jax.vmap(accumulate_clip_stats)(clip_range)
        new_successes, new_failures, new_rewards, new_episodes = new_stats
        
        # Update cumulative counts
        success_counts = self.clip_success_counts + new_successes
        failure_counts = self.clip_failure_counts + new_failures
        total_reward = self.clip_total_reward + new_rewards
        episode_counts = self.clip_episode_counts + new_episodes
        
        # Compute metrics for weighting
        # Success rate per clip (handle division by zero)
        success_rate = jnp.where(
            episode_counts > 0,
            success_counts / episode_counts,
            0.5  # Default for clips without data
        )
        
        # Average reward per clip
        avg_reward = jnp.where(
            episode_counts > 0,
            total_reward / episode_counts,
            0.0
        )
        
        # Compute raw weights based on configured strategy
        raw_weights = jnp.ones(self.num_clips)
        
        if self.use_success_weighting:
            # Weight harder clips (lower success rate) more heavily
            # Invert success rate so low success -> high weight
            inverse_success = 1.0 - success_rate
            raw_weights = raw_weights * (inverse_success + self.min_weight)
        
        if self.use_reward_weighting:
            # Weight clips with lower rewards more heavily
            # Normalize rewards to [0, 1] range first
            reward_range = jnp.max(avg_reward) - jnp.min(avg_reward)
            normalized_reward = jnp.where(
                reward_range > 1e-6,
                (avg_reward - jnp.min(avg_reward)) / reward_range,
                0.5
            )
            inverse_reward = 1.0 - normalized_reward
            raw_weights = raw_weights * (inverse_reward * self.reward_scale + self.min_weight)
        
        # Normalize to get probabilities
        raw_weights = jnp.maximum(raw_weights, self.min_weight)
        new_weights = raw_weights / jnp.sum(raw_weights)
        
        # Exponential moving average with previous weights
        # This prevents rapid changes and provides stability
        smoothed_weights = (
            self.ema_alpha * self.sampling_weights +
            (1 - self.ema_alpha) * new_weights
        )
        
        # Renormalize after smoothing
        final_weights = smoothed_weights / jnp.sum(smoothed_weights)
        
        return MotionClipWeightingState(
            num_clips=self.num_clips,
            clip_success_counts=success_counts,
            clip_failure_counts=failure_counts,
            clip_total_reward=total_reward,
            clip_episode_counts=episode_counts,
            sampling_weights=final_weights,
            min_weight=self.min_weight,
            ema_alpha=self.ema_alpha,
            reward_scale=self.reward_scale,
            use_success_weighting=self.use_success_weighting,
            use_reward_weighting=self.use_reward_weighting,
        )
    
    def get_metrics(self) -> dict[str, Array]:
        """Get metrics for logging."""
        success_rate = jnp.where(
            self.clip_episode_counts > 0,
            self.clip_success_counts / self.clip_episode_counts,
            0.0
        )
        
        avg_reward = jnp.where(
            self.clip_episode_counts > 0,
            self.clip_total_reward / self.clip_episode_counts,
            0.0
        )
        
        return {
            "motion_clip/weights_min": self.sampling_weights.min(),
            "motion_clip/weights_max": self.sampling_weights.max(),
            "motion_clip/weights_mean": self.sampling_weights.mean(),
            "motion_clip/weights_std": self.sampling_weights.std(),
            "motion_clip/success_rate_min": success_rate.min(),
            "motion_clip/success_rate_max": success_rate.max(),
            "motion_clip/success_rate_mean": success_rate.mean(),
            "motion_clip/avg_reward_min": avg_reward.min(),
            "motion_clip/avg_reward_max": avg_reward.max(),
            "motion_clip/avg_reward_mean": avg_reward.mean(),
            "motion_clip/total_episodes": self.clip_episode_counts.sum(),
        }


class WeightedInitialMotionStateReset(Reset):
    """Reset that samples motion clips using learned weights.
    
    This reset samples from motion clips with probabilities determined by
    the global weighting state, which is updated based on trajectory success.
    """
    
    freejoint: bool = False
    
    def __call__(
        self,
        data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
        shared_state: RolloutSharedState,  # Access global state
    ) -> tuple[PhysicsData, int]:
        """Reset to a random frame from a weighted-sampled motion clip.
        
        Args:
            data: Current physics data to reset
            curriculum_level: Current curriculum level
            rng: Random key
            shared_state: Global shared state containing motion clips and weights
            
        Returns:
            Tuple of (reset physics data, selected clip index)
        """
        # Get motion clips and weighting state from global state
        real_motions = shared_state.aux_values[REAL_MOTIONS_KEY]
        weighting_state = shared_state.aux_values[MOTION_CLIP_WEIGHTING_KEY]
        
        rng_clip, rng_frame = jax.random.split(rng)
        
        # Sample clip using learned weights
        clip_index = jax.random.choice(
            rng_clip,
            weighting_state.num_clips,
            p=weighting_state.sampling_weights
        )
        
        # Get the selected clip (assuming real_motions has shape [num_clips, T, ...])
        # This extracts clip_index from the batch dimension
        selected_clip = jax.tree_map(lambda x: x[clip_index], real_motions)
        
        # Sample a random frame from the selected clip
        num_frames = jax.tree_util.tree_leaves(selected_clip)[0].shape[0]
        frame_index = jax.random.randint(rng_frame, (), 0, num_frames)
        
        # Get qpos/qvel at the selected frame
        # Assuming selected_clip has qpos/qvel or is structured as MotionReferenceData
        # Adapt this based on your actual motion data structure
        qpos = jax.tree_util.tree_leaves(selected_clip)[0][frame_index]  # Simplified
        qvel = jax.tree_util.tree_leaves(selected_clip)[1][frame_index]  # Simplified
        
        # Update physics data
        if self.freejoint:
            data = update_data_field(data, "qpos", qpos)
            data = update_data_field(data, "qvel", qvel)
        else:
            # Keep root position/orientation, update joint positions
            new_qpos = jnp.concatenate([data.qpos[:7], qpos[7:]])
            new_qvel = jnp.concatenate([data.qvel[:6], qvel[7:]])
            data = update_data_field(data, "qpos", new_qpos)
            data = update_data_field(data, "qvel", new_qvel)
        
        # Return both reset data and the clip index (to track in trajectory)
        return data, clip_index


Config = TypeVar("Config", bound=AMPConfig)


class MotionClipWeightedAMPTask(AMPTask[Config], Generic[Config]):
    """AMP task with motion clip weighting based on trajectory success.
    
    This extends the standard AMP task to:
    1. Track which motion clip each environment uses
    2. Maintain global statistics on clip success rates
    3. Adjust sampling probabilities to focus on harder clips
    """
    
    def _get_shared_state(
        self,
        *,
        rng: PRNGKeyArray,
        mj_model: mujoco.MjModel,
        physics_model: PhysicsModel,
        model_arrs: tuple[PyTree, ...],
    ) -> RolloutSharedState:
        """Initialize shared state with motion clip weighting."""
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
        weighting_state = MotionClipWeightingState.create(
            num_clips=num_clips,
            min_weight=0.05,      # Minimum 5% probability per clip
            ema_alpha=0.95,       # Strong smoothing
            use_success_weighting=True,
            use_reward_weighting=False,
        )
        
        # Add to shared state
        shared_state = replace(
            shared_state,
            aux_values=xax.FrozenDict(
                shared_state.aux_values.unfreeze()
                | {MOTION_CLIP_WEIGHTING_KEY: weighting_state}
            ),
        )
        
        return shared_state
    
    def postprocess_trajectory(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
        trajectory: Trajectory,
        rng: PRNGKeyArray,
    ) -> Trajectory:
        """Add motion clip index to trajectory for tracking."""
        trajectory = super().postprocess_trajectory(
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
            trajectory=trajectory,
            rng=rng,
        )
        
        # NOTE: You need to implement tracking which clip was used during reset
        # This could be done by:
        # 1. Storing clip_index in env_states during reset
        # 2. Using a custom reset that returns the clip index
        # 3. Inferring from trajectory data
        
        # For now, this is a placeholder - you'll need to get the actual clip index
        # from wherever you store it during environment reset
        # clip_index = ...  # Get from env_states or command
        
        # aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
        # aux_outputs[MOTION_CLIP_INDEX_KEY] = clip_index
        # trajectory = replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))
        
        return trajectory
    
    def update_model(
        self,
        *,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array]]:
        """Update model and motion clip weights."""
        
        # Update motion clip statistics
        weighting_state = carry.shared_state.aux_values[MOTION_CLIP_WEIGHTING_KEY]
        
        # Get clip indices from trajectories
        # NOTE: This assumes you've added clip indices in postprocess_trajectory
        clip_indices = trajectories.aux_outputs.get(MOTION_CLIP_INDEX_KEY, None)
        
        if clip_indices is not None:
            # Check if episodes are done (termination at last timestep)
            episode_done = trajectories.done[..., -1]  # [num_envs]
            
            # Get success flags for completed episodes
            successes = trajectories.success[..., -1]  # [num_envs]
            
            # Compute episode returns
            returns = rewards.total.sum(axis=-1)  # Sum over time [num_envs]
            
            # Update weighting state
            new_weighting_state = weighting_state.update(
                clip_indices=clip_indices,
                episode_done=episode_done,
                successes=successes,
                returns=returns,
            )
            
            # Update carry with new weighting state
            carry = replace(
                carry,
                shared_state=replace(
                    carry.shared_state,
                    aux_values=xax.FrozenDict(
                        carry.shared_state.aux_values.unfreeze()
                        | {MOTION_CLIP_WEIGHTING_KEY: new_weighting_state}
                    ),
                ),
            )
        
        # Perform standard PPO/AMP model update
        carry, metrics = super().update_model(
            constants=constants,
            carry=carry,
            trajectories=trajectories,
            rewards=rewards,
            rng=rng,
        )
        
        # Add motion clip weighting metrics to logging
        if clip_indices is not None:
            weighting_metrics = new_weighting_state.get_metrics()
            metrics = xax.FrozenDict(metrics.unfreeze() | weighting_metrics)
        
        return carry, metrics


# Example usage in a task configuration:
"""
class MyWalkingTask(MotionClipWeightedAMPTask[MyConfig]):
    
    def get_resets(self, mj_model: mujoco.MjModel) -> Collection[Reset]:
        return [
            WeightedInitialMotionStateReset(freejoint=False),
            # ... other resets
        ]
    
    # ... implement other required methods
"""
