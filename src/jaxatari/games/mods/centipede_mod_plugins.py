from functools import partial
import time
from typing import Any

import chex
import jax
import jax.numpy as jnp

from jaxatari.games.jax_centipede import CentipedeState
from jaxatari.modification import JaxAtariInternalModPlugin
from jaxatari.wrappers import JaxatariWrapper

class SlowSpellMod(JaxAtariInternalModPlugin):
    """Player spells have a third the speed."""
    constants_overrides = {
        "PLAYER_SPELL_SPEED": 3,
    }

class RandomMushroomsMod(JaxAtariInternalModPlugin):
    """Initialize mushroom positions randomly."""
    """def __init__(self, env):
        super().__init__(env)
        self._env = env
        # Overrides initialize_mushroom_positions from env
        self._env.initialize_mushroom_positions = self.initialize_mushroom_positions.__get__(self._env)"""

    @partial(jax.jit, static_argnums=(0,))
    def initialize_mushroom_positions(self):
        # Overrides the default function from the env
        rows = jnp.arange(self._env.consts.MUSHROOM_NUMBER_OF_ROWS) # 19
        cols = jnp.arange(self._env.consts.MUSHROOM_NUMBER_OF_COLS) # 16
        key = jax.random.PRNGKey(time.time_ns() % (2 ** 32))
        p = 0.0888  # ~ 27 / 304 -> roughly same number of mushrooms on screen

        spawn = jax.random.bernoulli(key, p, (19,16))

        # --- Per-cell computation ---
        def cell_fn(row, col):
            row_is_even = (row % 2) == 0
            column_start = jnp.where(
                row_is_even,
                self._env.consts.MUSHROOM_COLUMN_START_EVEN,
                self._env.consts.MUSHROOM_COLUMN_START_ODD,
            )
            x = column_start + self._env.consts.MUSHROOM_X_SPACING * col
            y = row * self._env.consts.MUSHROOM_Y_SPACING + 7
            lives = jnp.where(spawn[row, col] != 0, 3, 0)
            return jnp.array([x, y, 0, lives], dtype=jnp.int32)

        # Vectorize across grid with nested vmaps
        grid = jax.vmap(lambda r: jax.vmap(lambda c: cell_fn(r, c))(cols))(rows)

        # Flatten to (N*M, 4)
        return grid.reshape(-1, 4)

class RandomPlayerMovementMod(JaxAtariInternalModPlugin):       # TODO: fix retrieval of action / no direct overwrite of step()
    """Overwrites player movement with random action with probability 0.2"""
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CentipedeState, action: int|float) -> tuple[chex.Array, CentipedeState, float, bool, dict[Any, Any]]:
        random_movement_key, random_action_key = jax.random.split(state.rng_key)

        random_movement_indicator = jax.random.bernoulli(random_movement_key,0.2)  # might be a bit heavy, lower also possible
        random_action = jax.random.randint(random_action_key, (), 0, 18)

        new_action = jnp.where(random_movement_indicator, random_action, action)
        return self._env.step(state, new_action)