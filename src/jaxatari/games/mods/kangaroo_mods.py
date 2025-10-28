import jax
import jax.numpy as jnp
import chex
from functools import partial


class KangarooEnvMod:
    """
    Mod that conditionally disables monkeys and/or falling coconuts via method replacement.
    """
    available_mods = ["no_monkey", "no_falling_coconut"]

    def __init__(self, env, mods_config: list = []):
        self._env = env        
        # --- Method Replacement Logic ---
        # 1. Disable Monkeys and Thrown Coconuts
        if "no_monkey" in mods_config:
            # Replace the original JITted method with the no-op JITted method
            self._env._monkey_controller = self._no_monkey_controller
            
        # 2. Disable Falling Coconut
        if "no_falling_coconut" in mods_config:
            # Replace the original JITted method with the no-op JITted method
            self._env._falling_coconut_controller = self._no_falling_coconut_controller
        
        for mod in mods_config:
            if mod not in self.available_mods:
                raise ValueError(f"Mod '{mod}' is not recognized. Available mods: {self.available_mods}")
            
    def __getattr__(self, name):
        """Delegates all attribute and method access to the wrapped environment."""
        return getattr(self._env, name)


    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _no_monkey_controller(self, state: chex.Array, punching: chex.Array):
        """
        No-op override for _monkey_controller: disables monkey spawning, movement, 
        thrown coconuts, and score gain.
        """
        # Return current (unchanged) state for all monkey/coco-related arrays
        # Score addition must be explicitly zero.
        score_addition = jnp.zeros((), dtype=jnp.int32)
        
        return (
            state.level.monkey_states,       # new_monkey_states (remains zeros)
            state.level.monkey_positions,    # new_monkey_positions (remains off-screen)
            state.level.monkey_throw_timers, # new_monkey_throw_timers (remains zeros)
            score_addition,                  # score_addition (0)
            state.level.coco_positions,      # new_coco_positions (remains off-screen)
            state.level.coco_states,         # new_coco_states (remains zeros)
            jnp.array(False),                # flip (should be False)
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _no_falling_coconut_controller(self, state: chex.Array, punching: chex.Array):
        """
        No-op override for _falling_coconut_controller: prevents the falling coconut logic.
        """
        # New state variables remain unchanged, preventing updates/spawning/score
        return (
            state.level.falling_coco_position,
            state.level.falling_coco_dropping,
            state.level.falling_coco_counter,
            state.level.falling_coco_skip_update,
            jnp.zeros((), dtype=jnp.int32),
        )