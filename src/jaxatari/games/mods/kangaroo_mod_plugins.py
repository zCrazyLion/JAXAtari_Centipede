import jax
import jax.numpy as jnp
import chex
from functools import partial

from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
from jaxatari.games.jax_kangaroo import KangarooState

# --- 1. Internal Mods (Group 1) ---

class NoMonkeyMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable monkeys.
    This patches the environment's '_monkey_controller' method.
    """
    
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _monkey_controller(self, state: KangarooState, punching: chex.Array):
        """
        No-op override for _monkey_controller.
        """
        score_addition = jnp.zeros((), dtype=jnp.int32)
        
        return (
            state.level.monkey_states,       
            state.level.monkey_positions,    
            state.level.monkey_throw_timers, 
            score_addition,                  
            state.level.coco_positions,      
            state.level.coco_states,         
            jnp.array(False),                
        )

class NoFallingCoconutMod(JaxAtariInternalModPlugin):
    """
    Internal mod to disable the single falling coconut.
    This patches the environment's '_falling_coconut_controller' method.
    """
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _falling_coconut_controller(self, 
                                    state: KangarooState, 
                                    punching: chex.Array
                                    ):
        """
        No-op override for _falling_coconut_controller.
        """
        return (
            state.level.falling_coco_position, 
            state.level.falling_coco_dropping, 
            state.level.falling_coco_counter,  
            state.level.falling_coco_skip_update,
            jnp.zeros((), dtype=jnp.int32),     
        )

# --- 2. Post-Step Mod (Group 2) ---

class PinChildMod(JaxAtariPostStepModPlugin):
    """
    Post-step mod to pin the child kangaroo in place.
    """
    @partial(jax.jit, static_argnums=(0,))
    def run(self, new_state: KangarooState):
        """
        Called *after* the main step. Overwrites the child's
        position with its static starting position.
        """
        # Get the level constants to find the child's start position
        level_constants = self._env._get_level_constants(new_state.current_level)
        
        # Pin the child's position and velocity
        pinned_level_state = new_state.level._replace(
            child_position=level_constants.child_position, #
            child_velocity=jnp.array(0) # Also stop its velocity
        )
        
        return new_state._replace(level=pinned_level_state)