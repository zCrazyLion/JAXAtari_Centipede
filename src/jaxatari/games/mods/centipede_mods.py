import functools
import time

import jax
import jax.numpy as jnp

from jaxatari.wrappers import JaxatariWrapper


class RandomMushrooms(JaxatariWrapper):
    """Initialize mushroom positions randomly."""
    def __init__(self, env):
        super().__init__(env)
        self._env = env
        # Overrides initialize_mushroom_positions from env
        self._env.initialize_mushroom_positions = self.initialize_mushroom_positions.__get__(self._env)

    @functools.partial(jax.jit, static_argnums=(0,))
    def initialize_mushroom_positions(self):
        # Overrides the default function from the env
        rows = jnp.arange(self.consts.MUSHROOM_NUMBER_OF_ROWS)
        cols = jnp.arange(self.consts.MUSHROOM_NUMBER_OF_COLS)
        key = jax.random.PRNGKey(time.time_ns() % (2 ** 32))
        p = 0.0888  # ~ 27 / 304 -> roughly same number of mushrooms on screen

        spawn = jax.random.bernoulli(key, p, (19,16))

        # --- Per-cell computation ---
        def cell_fn(row, col):
            row_is_even = (row % 2) == 0
            column_start = jnp.where(
                row_is_even,
                self.consts.MUSHROOM_COLUMN_START_EVEN,
                self.consts.MUSHROOM_COLUMN_START_ODD,
            )
            x = column_start + self.consts.MUSHROOM_X_SPACING * col
            y = row * self.consts.MUSHROOM_Y_SPACING + 7
            lives = jnp.where(spawn[row, col] != 0, 3, 0)
            return jnp.array([x, y, 0, lives], dtype=jnp.int32)

        # Vectorize across grid with nested vmaps
        grid = jax.vmap(lambda r: jax.vmap(lambda c: cell_fn(r, c))(cols))(rows)

        # Flatten to (N*M, 4)
        return grid.reshape(-1, 4)


