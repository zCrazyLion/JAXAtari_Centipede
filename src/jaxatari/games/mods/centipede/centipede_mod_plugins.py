import random
from functools import partial
import time
from typing import Any

import chex
import jax
import jax.numpy as jnp

from jaxatari.games.jax_centipede import CentipedeState
from jaxatari.games.jax_centipede import JaxCentipede
from jaxatari.modification import JaxAtariInternalModPlugin, JaxAtariPostStepModPlugin
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


_ORIGINAL_PLAYER_STEP = JaxCentipede.player_step

class RandomPlayerMovementMod(JaxAtariInternalModPlugin):
    """Overwrites player movement with a random action with probability RANDOM_ACTION_PROB."""

    RANDOM_ACTION_PROB: float = 0.5

    @partial(jax.jit, static_argnums=(0,))
    def player_step(
        self,
        player_x: chex.Array,
        player_y: chex.Array,
        player_velocity_x: chex.Array,
        action: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        # Fold traced values into the key so it varies every actual call,
        # instead of being a Python-level constant baked in at trace time.
        key = jax.random.PRNGKey(time.time_ns() % (2 ** 32))
        key = jax.random.fold_in(key, player_x.astype(jnp.int32))
        key = jax.random.fold_in(key, player_y.astype(jnp.int32))
        key = jax.random.fold_in(
            key, jax.lax.bitcast_convert_type(player_velocity_x.astype(jnp.float32), jnp.int32)
        )
        key = jax.random.fold_in(key, action.astype(jnp.int32))
        move_key, action_key = jax.random.split(key)

        use_random_action = jax.random.bernoulli(move_key, self.RANDOM_ACTION_PROB)
        random_action = jax.random.randint(action_key, (), 0, 18)
        new_action = jnp.where(use_random_action, random_action, action)

        # Call the pristine implementation directly — NOT self._env.player_step,
        # which is this very override and would recurse.
        return _ORIGINAL_PLAYER_STEP(self._env, player_x, player_y, player_velocity_x, new_action)