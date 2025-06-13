import jax
import jax.numpy as jnp
import pytest
from jaxatari.core import JAXAtari
from jaxatari.environment import EnvInfo, EnvObs, EnvState
from jaxatari.wrappers import (
    ObjectCentricWrapper,
    PixelObsWrapper,
    AtariWrapper,
    PixelAndObjectCentricWrapper,
    LogWrapper,
    MultiRewardLogWrapper
)
import numpy as np

def test_base_jaxatari():
    """Test the base JAXAtari class with a simple game."""
    env = JAXAtari("pong")
    key = jax.random.PRNGKey(0)
    
    # Test reset
    obs, state = env.reset(key)
    assert obs is not None
    assert state is not None
    
    # Test step
    action = 0  # NOOP
    obs, state, reward, done, info = env.step(state, action)
    assert obs is not None
    assert state is not None
    assert isinstance(reward, (float, jnp.ndarray))
    assert isinstance(done, (bool, jnp.ndarray))
    assert info is not None


def test_atari_wrapper():
    """Test the AtariWrapper."""
    base_env = JAXAtari("pong")
    env = AtariWrapper(base_env)
    key = jax.random.PRNGKey(0)
    
    # Test reset
    obs, state = env.reset(key)
    assert obs is not None
    assert state.env_state is not None
    assert state.step == 0
    assert state.prev_action == 0
    assert state.obs_stack is not None
    assert state.key is not None  # Check that key is stored in state
    
    # Test step
    action = 0  # NOOP
    obs, state, reward, done, info = env.step(state, action)
    assert obs is not None
    assert state.env_state is not None
    assert state.step > 0
    assert state.prev_action is not None
    assert state.obs_stack is not None
    assert state.key is not None  # Check that key is still in state
    assert isinstance(reward, (float, jnp.ndarray))
    assert isinstance(done, (bool, jnp.ndarray))
    assert info is not None

def test_obs_to_flat_array_with_stacked_observations():
    """Test that obs_to_flat_array works correctly with stacked observations."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = JAXAtari("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=4)
    env = ObjectCentricWrapper(atari_env)
    
    # Get initial observation
    obs, state = env.reset(key)

    # Verify shape matches expected size (frame_stack_size * obs_size)
    expected_size = base_env.env.obs_size * env.frame_stack_size
    assert obs.shape == (expected_size,), f"Expected shape ({expected_size},), got {obs.shape}"
    
    # Take a step and verify shape remains consistent
    obs, state, reward, done, info = env.step(state, 0)
    assert obs.shape == (expected_size,), f"Expected shape ({expected_size},), got {obs.shape}"
    
    # Verify the observation contains the expected number of frames worth of data
    # This is a basic check - we might want to add more specific checks based on the game's observation structure
    assert obs.size == expected_size, f"Expected {expected_size} elements, got {obs.size}"

def test_pixel_obs_wrapper_with_stacked_frames():
    """Test that PixelObsWrapper correctly handles stacked frames."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = JAXAtari("pong")
    env = PixelObsWrapper(AtariWrapper(base_env))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify shape is (frame_stack_size, height, width, channels)
    # Pong dimensions are 160x210 with 3 color channels
    expected_shape = (env.frame_stack_size, 160, 210, 3)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    
    # Take a step and verify shape remains consistent
    obs, state, reward, done, info = env.step(state, 0)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"

    # perform 100 steps to get to a state in which the frames should be different
    for _ in range(100):
        obs, state, reward, done, info = env.step(state, 0)
    
    # Verify that frames are different (not just copies)
    # The first and last frame in the stack should be different
    # This is a basic check - in practice, they might be similar if the game state hasn't changed much
    assert not jnp.array_equal(obs[0], obs[-1]), "First and last frames should be different"
    
    # Verify that frames are in the correct range (0-255 for uint8)
    assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"

def test_pixel_and_object_centric_wrapper():
    """Test that PixelAndObjectCentricWrapper correctly handles both image and object observations."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = JAXAtari("pong")
    env = PixelAndObjectCentricWrapper(AtariWrapper(base_env))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify shape of image part (frame_stack_size, height, width, channels)
    image_obs = obs[0]
    expected_image_shape = (env.frame_stack_size, 160, 210, 3)
    assert image_obs.shape == expected_image_shape, f"Expected image shape {expected_image_shape}, got {image_obs.shape}"
    
    # Verify shape of object-centric part (frame_stack_size * obs_size)
    object_obs = obs[1]
    expected_object_shape = (env.frame_stack_size * base_env.env.obs_size,)
    assert object_obs.shape == expected_object_shape, f"Expected object shape {expected_object_shape}, got {object_obs.shape}"
    
    # Take a step and verify shapes remain consistent
    obs, state, reward, done, info = env.step(state, 0)
    assert obs[0].shape == expected_image_shape
    assert obs[1].shape == expected_object_shape
    assert isinstance(reward, (float, jnp.ndarray))
    assert isinstance(done, (bool, jnp.ndarray))
    assert info is not None

def test_object_centric_wrapper_flattens_stacked_observations():
    """Test that ObjectCentricWrapper correctly flattens stacked observations."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = JAXAtari("pong")
    atari_env = AtariWrapper(base_env, frame_stack_size=4)
    env = ObjectCentricWrapper(atari_env)
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Get the size of a single observation
    single_obs_size = base_env.env.obs_size
    
    # Verify shape matches expected size (frame_stack_size * obs_size)
    expected_size = single_obs_size * env.frame_stack_size
    assert obs.shape == (expected_size,), f"Expected shape ({expected_size},), got {obs.shape}"
    
    # Take a step and verify shape remains consistent
    obs, state, reward, done, info = env.step(state, 0)
    assert obs.shape == (expected_size,), f"Expected shape ({expected_size},), got {obs.shape}"

    # run warmup for 100 steps and then check that the frames and obs are different
    for _ in range(100):
        obs, state, reward, done, info = env.step(state, 0)
    
    assert not jnp.array_equal(obs[0], obs[-1]), "First and last frames should be different"
    
    # Verify the observation contains the expected number of frames worth of data
    assert obs.size == expected_size, f"Expected {expected_size} elements, got {obs.size}"
    
    # Verify that the flattened array contains the correct number of frames
    # Each frame's data should be contiguous in the flattened array
    for i in range(env.frame_stack_size):
        frame_start = i * single_obs_size
        frame_end = (i + 1) * single_obs_size
        frame_data = obs[frame_start:frame_end]
        assert frame_data.size == single_obs_size, f"Frame {i} has incorrect size"

def test_log_wrapper():
    """Test that LogWrapper correctly tracks episode returns and lengths."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = JAXAtari("pong")
    env = LogWrapper(PixelObsWrapper(AtariWrapper(base_env)))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify initial state
    assert state.episode_returns == 0.0
    assert state.episode_lengths == 0
    assert state.returned_episode_returns == 0.0
    assert state.returned_episode_lengths == 0
    
    # Verify observation format (should match PixelObsWrapper)
    expected_shape = (env.frame_stack_size, 160, 210, 3)
    assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
    assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"
    
    # Take some steps and accumulate rewards
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done and steps < 100:  # Limit steps to avoid infinite loops
        obs, state, reward, done, info = env.step(state, 0)  # Use NOOP action
        total_reward += reward
        steps += 1
        
        # Verify observation format remains consistent
        assert obs.shape == expected_shape, f"Expected shape {expected_shape}, got {obs.shape}"
        assert jnp.all(obs >= 0) and jnp.all(obs <= 255), "Pixel values should be in range [0, 255]"
        
        # Verify running totals
        assert state.episode_returns == total_reward * (1 - done)
        assert state.episode_lengths == steps * (1 - done)
        
        if done:
            # Verify final episode statistics
            assert state.returned_episode_returns == total_reward
            assert state.returned_episode_lengths == steps
            assert info["returned_episode_returns"] == total_reward
            assert info["returned_episode_lengths"] == steps
            assert info["returned_episode"] == True

def test_multi_reward_log_wrapper():
    """Test that MultiRewardLogWrapper correctly tracks multiple reward types."""
    key = jax.random.PRNGKey(0)
    
    # Create environment with wrappers
    base_env = JAXAtari("pong")
    env = MultiRewardLogWrapper(PixelAndObjectCentricWrapper(AtariWrapper(base_env)))
    
    # Get initial observation
    obs, state = env.reset(key)
    
    # Verify initial state
    assert state.episode_returns_env == 0.0
    assert jnp.all(state.episode_returns == 0.0)
    assert state.episode_lengths == 0
    assert state.returned_episode_returns_env == 0.0
    assert jnp.all(state.returned_episode_returns == 0.0)
    assert state.returned_episode_lengths == 0
    
    # Verify observation format (should match PixelAndObjectCentricWrapper)
    image_obs, object_obs = obs
    expected_image_shape = (env.frame_stack_size, 160, 210, 3)
    expected_object_shape = (env.frame_stack_size * base_env.env.obs_size,)
    assert image_obs.shape == expected_image_shape, f"Expected image shape {expected_image_shape}, got {image_obs.shape}"
    assert object_obs.shape == expected_object_shape, f"Expected object shape {expected_object_shape}, got {object_obs.shape}"
    assert jnp.all(image_obs >= 0) and jnp.all(image_obs <= 255), "Pixel values should be in range [0, 255]"
    
    # Take some steps and accumulate rewards
    total_reward_env = 0.0
    total_rewards = jnp.zeros_like(state.episode_returns)
    steps = 0
    done = False
    
    while not done and steps < 100:  # Limit steps to avoid infinite loops
        obs, state, reward, done, info = env.step(state, 0)  # Use NOOP action
        total_reward_env += reward
        total_rewards += info["all_rewards"]
        steps += 1
        
        # Verify observation format remains consistent
        image_obs, object_obs = obs
        assert image_obs.shape == expected_image_shape, f"Expected image shape {expected_image_shape}, got {image_obs.shape}"
        assert object_obs.shape == expected_object_shape, f"Expected object shape {expected_object_shape}, got {object_obs.shape}"
        assert jnp.all(image_obs >= 0) and jnp.all(image_obs <= 255), "Pixel values should be in range [0, 255]"
        
        # Verify running totals
        assert state.episode_returns_env == total_reward_env * (1 - done)
        assert jnp.all(state.episode_returns == total_rewards * (1 - done))
        assert state.episode_lengths == steps * (1 - done)
        
        if done:
            # Verify final episode statistics
            assert state.returned_episode_returns_env == total_reward_env
            assert jnp.all(state.returned_episode_returns == total_rewards)
            assert state.returned_episode_lengths == steps
            assert info["returned_episode_env_returns"] == total_reward_env
            for i, r in enumerate(total_rewards):
                assert info[f"returned_episode_returns_{i}"] == r
            assert info["returned_episode_lengths"] == steps
            assert info["returned_episode"] == True

if __name__ == "__main__":
    pytest.main([__file__]) 