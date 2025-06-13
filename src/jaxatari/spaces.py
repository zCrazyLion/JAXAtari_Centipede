# Copyright 2020 Rémi Louf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is derived from the Gymnax project (https://github.com/RobertTLange/gymnax)
# and has been modified for the purposes of this project.

import collections
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

class Space:
    """Minimal jittable class for abstract spaces."""

    def sample(self, key: jax.Array) -> jax.Array:
        raise NotImplementedError

    def contains(self, x: jax.Array) -> Any:
        raise NotImplementedError
    
    '''
    Returns the range of the space with the first value being the minimum and the second value being the maximum.
    Only implemented for numerically bounded spaces.
    '''
    def range(self):
        raise NotImplementedError


class Discrete(Space):
    """Minimal jittable class for discrete spaces."""

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int32

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            key, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        x = x.astype(jnp.int32)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond
    
    def range(self) -> tuple[float, float]:
        return 0, self.n - 1


class Box(Space):
    """Minimal jittable class for array-shaped spaces."""

    def __init__(
        self,
        low: jnp.ndarray | float,
        high: jnp.ndarray | float,
        shape: Any,  # Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            key, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jax.Array) -> jax.Array:
        """Check whether specific object is within space."""
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond
    
    def range(self) -> tuple[float, float]:
        return self.low, self.high


class Dict(Space):
    """Minimal jittable class for dictionary of simpler jittable spaces."""

    def __init__(self, spaces: Any):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, key: jax.Array) -> Any:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(key, self.num_spaces)
        return collections.OrderedDict(
            [
                (k, self.spaces[k].sample(key_split[i]))
                for i, k in enumerate(self.spaces)
            ]
        )

    def contains(self, x: jax.Array) -> bool:
        """Check whether dimensions of object are within subspace."""
        # Check for each space individually
        out_of_space = 0
        for k, space in self.spaces.items():
            out_of_space += 1 - space.contains(getattr(x, k)).astype(jnp.int32)
        return out_of_space == 0
    
    def range(self) -> tuple[tuple[float, float], ...]:
        return tuple(space.range() for space in self.spaces.values())


class Tuple(Space):
    """Minimal jittable class for tuple (product) of jittable spaces."""

    def __init__(self, spaces: Sequence[Space]):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, key: jax.Array) -> Any:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(key, self.num_spaces)
        return tuple([s.sample(key_split[i]) for i, s in enumerate(self.spaces)])

    def contains(self, x: jax.Array) -> bool:
        """Check whether dimensions of object are within subspace."""
        # Check for each space individually
        out_of_space = 0
        for i, space in enumerate(self.spaces):
            out_of_space += 1 - space.contains(x[i])
        return out_of_space == 0
    
    def range(self) -> tuple[tuple[float, float], ...]:
        return tuple(space.range() for space in self.spaces)