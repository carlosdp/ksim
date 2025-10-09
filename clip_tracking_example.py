"""
Example of tracking motion clip indices through the system.

The main challenge is: how do we know which motion clip each environment used?
This file shows several approaches.
"""

from dataclasses import dataclass, replace
from typing import Collection

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.commands import Command
from ksim.resets import Reset
from ksim.task.rl import RolloutEnvState, RolloutSharedState
from ksim.types import PhysicsData, Trajectory
from ksim.utils.mujoco import update_data_field
from ksim.utils.priors import MotionReferenceData


# ============================================================================
# APPROACH 1: Store in Environment State (Simplest)
# ============================================================================

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ExtendedRolloutEnvState(RolloutEnvState):
    """Extended environment state that tracks the current motion clip."""
    current_motion_clip_index: Array  # Scalar int per environment


# In your task, you would override the state creation to include this field
# But this requires modifying core framework code, so not recommended


# ============================================================================
# APPROACH 2: Store in Trajectory aux_outputs (Recommended)
# ============================================================================

class MotionClipTrackingReset(Reset):
    """Reset that tracks which clip was used.
    
    This is the simplest approach: the reset function determines which clip
    to use, applies it, and returns the clip index. Then we store it in the
    trajectory during environment stepping.
    """
    
    reference_motions: PyTree  # Motion clips [num_clips, T, ...]
    freejoint: bool = False
    
    def __call__(
        self, 
        data: PhysicsData, 
        curriculum_level: Array, 
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, dict[str, Array]]:
        """Reset and return metadata about which clip was used.
        
        Returns:
            Tuple of (reset physics data, metadata dict with clip index)
        """
        num_clips = jax.tree_util.tree_leaves(self.reference_motions)[0].shape[0]
        
        rng_clip, rng_frame = jax.random.split(rng)
        
        # Sample clip uniformly (or use weights from shared state)
        clip_index = jax.random.randint(rng_clip, (), 0, num_clips)
        
        # Get selected clip
        selected_clip = jax.tree_map(lambda x: x[clip_index], self.reference_motions)
        
        # Sample frame
        num_frames = jax.tree_util.tree_leaves(selected_clip)[0].shape[0]
        frame_index = jax.random.randint(rng_frame, (), 0, num_frames)
        
        # Apply to physics data (simplified - adapt to your motion structure)
        qpos = jax.tree_util.tree_leaves(selected_clip)[0][frame_index]
        qvel = jax.tree_util.tree_leaves(selected_clip)[1][frame_index]
        
        if self.freejoint:
            data = update_data_field(data, "qpos", qpos)
            data = update_data_field(data, "qvel", qvel)
        else:
            new_qpos = jnp.concatenate([data.qpos[:7], qpos[7:]])
            new_qvel = jnp.concatenate([data.qvel[:6], qvel[7:]])
            data = update_data_field(data, "qpos", new_qpos)
            data = update_data_field(data, "qvel", new_qvel)
        
        # Return clip index in metadata
        metadata = {"motion_clip_index": clip_index}
        return data, metadata


# Then in your environment step function, you would store this in aux_outputs
# This requires modifying the step function to accept and propagate metadata


# ============================================================================
# APPROACH 3: Store in Command (Most Integrated)
# ============================================================================

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MotionClipCommandValue:
    """Command that tracks which motion clip to use."""
    clip_index: Array  # Scalar int


@attrs.define(frozen=True, kw_only=True)
class MotionClipCommand(Command):
    """Command that selects and tracks motion clips.
    
    This approach integrates clip selection into the command system,
    making it easy to track which clip each environment is using.
    """
    
    num_clips: int
    switch_prob: float = attrs.field(default=0.0)  # Probability of changing clip
    
    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> MotionClipCommandValue:
        """Initialize with a random clip."""
        clip_index = jax.random.randint(rng, (), 0, self.num_clips)
        return MotionClipCommandValue(clip_index=clip_index)
    
    def __call__(
        self,
        prev_command: MotionClipCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> MotionClipCommandValue:
        """Potentially switch to a new clip."""
        rng_switch, rng_clip = jax.random.split(rng)
        
        switch_mask = jax.random.bernoulli(rng_switch, self.switch_prob)
        new_clip_index = jax.random.randint(rng_clip, (), 0, self.num_clips)
        
        clip_index = jnp.where(switch_mask, new_clip_index, prev_command.clip_index)
        
        return MotionClipCommandValue(clip_index=clip_index)
    
    def get_metrics(self, command: MotionClipCommandValue, physics_data: PhysicsData) -> xax.FrozenDict[str, Array]:
        """Log current clip index."""
        return xax.FrozenDict({"motion_clip_index": command.clip_index})


# Usage in task:
"""
class MyTask(AMPTask):
    def get_commands(self, mj_model):
        num_clips = self.get_real_motions(mj_model).shape[0]
        return [
            MotionClipCommand(num_clips=num_clips),
            # ... other commands
        ]
"""

# Then in postprocess_trajectory:
"""
def postprocess_trajectory(...) -> Trajectory:
    trajectory = super().postprocess_trajectory(...)
    
    # Extract clip index from command
    clip_index = trajectory.command["motion_clip_command"].clip_index
    
    # Add to aux_outputs
    aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
    aux_outputs["motion_clip_index"] = clip_index
    
    return replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))
"""


# ============================================================================
# APPROACH 4: Infer from Reset Data (Hacky but Works)
# ============================================================================

class MotionClipReset(Reset):
    """Reset that stores clip index in a specific qpos element."""
    
    reference_motions: PyTree
    freejoint: bool = False
    storage_index: int = -1  # Which qpos element to (ab)use for storage
    
    def __call__(
        self,
        data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PhysicsData:
        """Reset and encode clip index in qpos."""
        num_clips = jax.tree_util.tree_leaves(self.reference_motions)[0].shape[0]
        
        rng_clip, rng_frame = jax.random.split(rng)
        clip_index = jax.random.randint(rng_clip, (), 0, num_clips)
        
        # ... apply reset as normal ...
        
        # Store clip index in unused qpos element (HACKY!)
        # This could be a joint that's always zero, or add a dummy joint
        qpos = data.qpos.at[self.storage_index].set(clip_index.astype(jnp.float32))
        data = update_data_field(data, "qpos", qpos)
        
        return data


# Then extract in postprocess:
"""
def postprocess_trajectory(...) -> Trajectory:
    # Extract from qpos
    clip_index = trajectory.qpos[:, 0, STORAGE_INDEX].astype(jnp.int32)
    
    aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
    aux_outputs["motion_clip_index"] = clip_index
    
    return replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))
"""


# ============================================================================
# APPROACH 5: Use Event System (Clean but Complex)
# ============================================================================

from ksim.events import Event

class MotionClipSelectionEvent(Event):
    """Event that fires when a motion clip is selected.
    
    This uses the event system to track state changes.
    """
    
    def initial_state(self, rng: PRNGKeyArray) -> Array:
        """Initialize with clip 0."""
        return jnp.array(0, dtype=jnp.int32)
    
    def __call__(
        self,
        physics_data: PhysicsData,
        prev_state: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """State is the current clip index."""
        # This would be called during reset to update the clip
        # Implementation depends on your event triggering logic
        return prev_state


# ============================================================================
# RECOMMENDED APPROACH: Command + Postprocess
# ============================================================================

"""
The cleanest approach is:

1. Use MotionClipCommand to track which clip each environment should use
2. Access this in your reset logic via trajectory.command
3. Store in trajectory.aux_outputs during postprocess_trajectory
4. Update weights in update_model using aux_outputs

Full example:
"""

class RecommendedImplementation:
    """
    class MyAMPTask(AMPTask):
        
        def get_commands(self, mj_model):
            num_clips = len(self.get_real_motions(mj_model))
            return [
                MotionClipCommand(
                    num_clips=num_clips,
                    switch_prob=0.1,  # 10% chance to switch clip each episode
                ),
                # ... other commands
            ]
        
        def postprocess_trajectory(
            self,
            constants: RolloutConstants,
            env_states: RolloutEnvState,
            shared_state: RolloutSharedState,
            trajectory: Trajectory,
            rng: PRNGKeyArray,
        ) -> Trajectory:
            trajectory = super().postprocess_trajectory(...)
            
            # Extract clip index from command
            # The command is stored per timestep, so take first timestep
            clip_cmd: MotionClipCommandValue = trajectory.command["motion_clip_command"]
            clip_index = clip_cmd.clip_index[0]  # Same for all timesteps in episode
            
            # Store in aux_outputs for use in update_model
            aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
            aux_outputs["motion_clip_index"] = clip_index
            
            return replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))
        
        def update_model(self, ...) -> ...:
            # Now we can access clip indices
            clip_indices = trajectories.aux_outputs["motion_clip_index"]  # [num_envs]
            
            # Update weighting state
            weighting_state = carry.shared_state.aux_values["motion_clip_weighting"]
            new_state = weighting_state.update(
                clip_indices=clip_indices,
                successes=trajectories.success[..., -1],
                returns=rewards.total.sum(axis=-1),
            )
            
            # ... continue with update
    """


# ============================================================================
# WEIGHTED SAMPLING USING COMMAND
# ============================================================================

@attrs.define(frozen=True, kw_only=True)
class WeightedMotionClipCommand(Command):
    """Command that samples clips using learned weights from global state."""
    
    num_clips: int
    switch_prob: float = attrs.field(default=0.0)
    
    def _get_weights(self, shared_state: RolloutSharedState | None) -> Array:
        """Get sampling weights from global state, or uniform if not available."""
        if shared_state is not None and "motion_clip_weighting" in shared_state.aux_values:
            weighting_state = shared_state.aux_values["motion_clip_weighting"]
            return weighting_state.sampling_weights
        else:
            # Fallback to uniform
            return jnp.ones(self.num_clips) / self.num_clips
    
    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
        shared_state: RolloutSharedState | None = None,
    ) -> MotionClipCommandValue:
        """Initialize with weighted sampling."""
        weights = self._get_weights(shared_state)
        clip_index = jax.random.choice(rng, self.num_clips, p=weights)
        return MotionClipCommandValue(clip_index=clip_index)
    
    def __call__(
        self,
        prev_command: MotionClipCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
        shared_state: RolloutSharedState | None = None,
    ) -> MotionClipCommandValue:
        """Switch clips with weighted sampling."""
        rng_switch, rng_clip = jax.random.split(rng)
        
        switch_mask = jax.random.bernoulli(rng_switch, self.switch_prob)
        weights = self._get_weights(shared_state)
        new_clip_index = jax.random.choice(rng_clip, self.num_clips, p=weights)
        
        clip_index = jnp.where(switch_mask, new_clip_index, prev_command.clip_index)
        
        return MotionClipCommandValue(clip_index=clip_index)


"""
NOTE: The Command.__call__ interface doesn't natively support shared_state,
so you may need to modify the command update logic in the framework, or
access shared_state through a different mechanism.

A simpler approach is to update clip selection during environment reset
rather than during command updates, since reset has access to shared_state.
"""
