import jax
import jax.numpy as jnp
import time
from functools import partial
from jax_pong import Game as JaxPong, State
from jax_seaquest import Game as JaxSeaquest


def run_parallel_envs(num_steps: int = 1_000_000, num_envs: int = 2000):
    # Set device to GPU if available
    jax.default_device = jax.devices("gpu")[0] if len(jax.devices("gpu")) > 0 else jax.devices("cpu")[0]

    # Initialize environment
    env = JaxPong(frameskip=1)

    # Create a single environment reset for reference
    reset_fn = jax.jit(env.reset)

    # Create batched state from broadcasting a single reset
    init_state = reset_fn()
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_envs), init_state)

    # Vectorize the step function across environments
    @partial(jax.vmap, in_axes=(0, 0))
    def parallel_step(states, actions):
        return env.step(states, actions)

    # JIT compile the parallel step
    jit_parallel_step = jax.jit(parallel_step)

    # Initialize random key for action sampling
    rng_key = jax.random.PRNGKey(0)

    @jax.jit
    def run_one_step(carry, _):
        states, rng_key = carry

        # Split RNG key for action sampling
        rng_key, action_key = jax.random.split(rng_key)

        # Sample random actions for all environments
        actions = jax.random.randint(action_key,
                                     shape=(num_envs,),
                                     minval=0,
                                     maxval=6)  # 6 possible actions in Pong

        # Step all environments forward
        next_states = jit_parallel_step(states, actions)

        return (next_states, rng_key), None

    # Calculate number of steps per environment to reach total steps
    steps_per_env = num_steps // num_envs

    # Start timing
    start_time = time.time()

    # Run parallel simulation using scan
    (final_states, _), _ = jax.lax.scan(
        run_one_step,
        (states, rng_key),
        None,
        length=steps_per_env
    )

    # Calculate metrics
    total_time = time.time() - start_time
    total_steps = steps_per_env * num_envs
    steps_per_second = total_steps / total_time

    return total_time, steps_per_second, steps_per_env * num_envs


if __name__ == "__main__":
    # Print device information
    print("\nDevice Information:")
    print("Available devices:", jax.devices())
    print("Default device:", jax.default_device())

    # Run benchmark
    print("\nRunning parallel environment simulation...")
    total_time, steps_per_second, total_steps = run_parallel_envs()

    print(f"\nBenchmark Results:")
    print(f"Total steps completed: {total_steps:,}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average steps per second: {steps_per_second:,.2f}")
    print(f"Microseconds per step: {(total_time * 1_000_000 / total_steps):.2f}")