"""Tests for empirical normalization in PPO."""

import jax.numpy as jnp
import pytest

from ksim.task.ppo import (
    EmpiricalNormalizationState,
    apply_empirical_normalization,
    update_empirical_normalization_state,
)


def test_empirical_normalization_state_initialization():
    """Test that we can create an empirical normalization state."""
    state = EmpiricalNormalizationState(
        mean=jnp.zeros(10),
        var=jnp.ones(10),
        count=jnp.array(0.0),
    )
    assert state.mean.shape == (10,)
    assert state.var.shape == (10,)
    assert state.count.shape == ()


def test_update_empirical_normalization_state():
    """Test that updating the normalization state works correctly."""
    # Initialize state
    state = EmpiricalNormalizationState(
        mean=jnp.zeros(3),
        var=jnp.ones(3),
        count=jnp.array(0.0),
    )
    
    # Create some test data
    data = jnp.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ])
    
    # Update state
    new_state = update_empirical_normalization_state(state, data)
    
    # Check that statistics are updated
    assert new_state.count == 3.0
    expected_mean = jnp.array([2.0, 3.0, 4.0])
    assert jnp.allclose(new_state.mean, expected_mean, atol=1e-6)
    
    # Variance should be calculated correctly
    expected_var = jnp.array([2.0/3.0, 2.0/3.0, 2.0/3.0])
    assert jnp.allclose(new_state.var, expected_var, atol=1e-6)


def test_update_empirical_normalization_state_incremental():
    """Test that incremental updates work correctly."""
    # Initialize state
    state = EmpiricalNormalizationState(
        mean=jnp.array([1.0, 2.0]),
        var=jnp.array([1.0, 1.0]),
        count=jnp.array(10.0),
    )
    
    # Add more data
    data = jnp.array([[2.0, 3.0], [3.0, 4.0]])
    new_state = update_empirical_normalization_state(state, data)
    
    # Check count is updated
    assert new_state.count == 12.0
    
    # Mean should shift towards the new data
    assert jnp.all(new_state.mean > state.mean)


def test_apply_empirical_normalization():
    """Test that normalization is applied correctly."""
    # Create a state with known mean and variance
    state = EmpiricalNormalizationState(
        mean=jnp.array([2.0, 4.0]),
        var=jnp.array([1.0, 4.0]),
        count=jnp.array(100.0),
    )
    
    # Create test data
    data = jnp.array([[3.0, 6.0], [1.0, 2.0]])
    
    # Apply normalization
    normalized = apply_empirical_normalization(data, state)
    
    # Check shapes match
    assert normalized.shape == data.shape
    
    # Check normalization is correct
    # (3.0 - 2.0) / sqrt(1.0) = 1.0
    # (6.0 - 4.0) / sqrt(4.0) = 1.0
    assert jnp.allclose(normalized[0, 0], 1.0, atol=1e-6)
    assert jnp.allclose(normalized[0, 1], 1.0, atol=1e-6)
    
    # (1.0 - 2.0) / sqrt(1.0) = -1.0
    # (2.0 - 4.0) / sqrt(4.0) = -1.0
    assert jnp.allclose(normalized[1, 0], -1.0, atol=1e-6)
    assert jnp.allclose(normalized[1, 1], -1.0, atol=1e-6)


def test_apply_empirical_normalization_with_batches():
    """Test normalization with multiple batch dimensions."""
    state = EmpiricalNormalizationState(
        mean=jnp.array([0.0, 0.0]),
        var=jnp.array([1.0, 1.0]),
        count=jnp.array(100.0),
    )
    
    # Create data with shape (batch, time, feature)
    data = jnp.ones((4, 5, 2))
    
    # Apply normalization
    normalized = apply_empirical_normalization(data, state)
    
    # All values should be 1.0 (since (1 - 0) / sqrt(1) = 1)
    assert normalized.shape == data.shape
    assert jnp.allclose(normalized, 1.0, atol=1e-6)


def test_normalization_numerical_stability():
    """Test that normalization handles edge cases."""
    state = EmpiricalNormalizationState(
        mean=jnp.array([0.0]),
        var=jnp.array([0.0]),  # Zero variance edge case
        count=jnp.array(1.0),
    )
    
    data = jnp.array([[1.0], [2.0]])
    
    # Should not produce NaN or Inf due to eps parameter
    normalized = apply_empirical_normalization(data, state, eps=1e-8)
    assert jnp.all(jnp.isfinite(normalized))
