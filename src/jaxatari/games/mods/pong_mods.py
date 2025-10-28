import jax
import jax.numpy as jnp
import chex
from functools import partial


class PongEnvMod:
    """
    Mods for Pong.
    """
    available_mods = ["lazy_enemy", "_random_enemy_step"]

    def __init__(self, env, mods_config: list = []):
        self._env = env        
        # --- Method Replacement Logic ---
        # 1. Disable Monkeys and Thrown Coconuts
        if "lazy_enemy" in mods_config:
            # Replace the original JITted method with the no-op JITted method
            self._env._enemy_step = self._lazy_enemy_step
            
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
    def _lazy_enemy_step(self, state, step_counter, ball_y, ball_speed_y):
        should_move = (state.step_counter % 8 != 0) & (state.ball_vel_x < 0)

        direction = jnp.sign(ball_y - state.enemy_y)

        new_y = state.enemy_y + (direction * self.consts.ENEMY_STEP_SIZE).astype(jnp.int32)
        return jax.lax.cond(
            should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _random_enemy_step(self, state, step_counter, ball_y, ball_speed_y) -> PongState:
        random_cond = step_counter % 3 == 0  # make other move every 3 steps
        new_enemy_y = jnp.where(
            random_cond,
            prev_state.enemy_y - state.enemy_speed, # Move in the opposite direction than it would normally
            state.enemy_y  # Otherwise, use the current enemy's y position
        )
        new_enemy_speed = jnp.where(
            random_cond,
            -state.enemy_speed,  # Reverse the enemy's speed if the condition is met
            state.enemy_speed  # Otherwise, keep the current enemy's speed
        )

        new_state = state._replace(
            enemy_y = new_enemy_y,
            enemy_speed = new_enemy_speed,
        )

        return new_state