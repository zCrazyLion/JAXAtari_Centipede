import functools
from typing import Any, Dict, Tuple, Union


import chex
import jax
import jax.numpy as jnp
from jaxatari.games.jax_breakout import BreakoutState

from jaxatari.wrappers import JaxatariWrapper

class LeftDrift(JaxatariWrapper):
    """Drift the ball and paddles to the left"""
    def __init__(self, env):
        super().__init__(env)
        self.WALL_SIDE_WIDTH = 8
        self.BALL_SIZE = (2, 4)  # Width, Height of ball

    @functools.partial(jax.jit, static_argnums=(0,))
    def left_drift(self, prev_state: BreakoutState, state: BreakoutState) -> BreakoutState:
        side_collision = jnp.logical_or(
            state.ball_x <= self.WALL_SIDE_WIDTH+10, # left
            state.ball_x >= 170 - self.WALL_SIDE_WIDTH - self.BALL_SIZE[0]  # right
        )
        in_bounds_x = jnp.logical_not(side_collision)

        new_ball_x = jnp.where(
            in_bounds_x,
            state.ball_x-1,
            state.ball_x,
        )
        new_state = state._replace(
            ball_x = new_ball_x,
        )
        return new_state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: BreakoutState, action: Union[int, float]) -> Tuple[chex.Array, BreakoutState, float, bool, Dict[Any, Any]]:
        new_obs, new_state, reward, done, info = self._env.step(state, action)
        new_state = self.left_drift(state, new_state)

        return new_obs, new_state, reward, done, info