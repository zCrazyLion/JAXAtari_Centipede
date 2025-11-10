import jax
import jax.numpy as jnp
from functools import partial

from jaxatari.modification import JaxAtariPostStepModPlugin, JaxAtariInternalModPlugin
from jaxatari.games.jax_freeway import FreewayState


class StopAllCarsMod(JaxAtariPostStepModPlugin):
    """Stops all cars randomly with probability 0.4"""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        key = jax.random.PRNGKey(new_state.time)
        chance = 0.4
        random_bool = jax.random.bernoulli(key, chance)
        
        new_cars = jax.lax.cond(
            random_bool,
            lambda _: prev_state.cars,  # Keep the previous cars' x positions if the random condition is met
            lambda _: new_state.cars,  # Otherwise, use the current cars' x positions
            operand=None
        )
        
        return new_state._replace(cars=new_cars)


class AlwaysStopAllCarsMod(JaxAtariPostStepModPlugin):
    """Stops all cars after spawning"""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        # Always keep the previous cars' x positions
        return new_state._replace(cars=prev_state.cars)


class SpeedModeMod(JaxAtariPostStepModPlugin):
    """Increase speed of all cars by a factor of 2"""
    
    @partial(jax.jit, static_argnums=(0,))
    def run(self, prev_state: FreewayState, new_state: FreewayState) -> FreewayState:
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        
        To double speed, we look one step ahead from the current new_state.
        This is equivalent to calling step(new_state, 0) and taking the cars.
        """
        # Simulate one more step forward to get double speed
        # We need to check if cars should move at new_state.time (which is prev_state.time + 1)
        # and apply that movement to the already-updated cars
        fast_cars = new_state.cars
        for lane in range(self._env.consts.num_lanes):
            dir = (
                self._env.consts.car_update[lane] / jnp.abs(self._env.consts.car_update[lane])
            ).astype(jnp.int32)
            
            # Check if car should move at new_state.time (one step ahead)
            should_move = jnp.equal(
                jnp.mod(new_state.time, self._env.consts.car_update[lane]), 0
            )
            
            new_x = jax.lax.cond(
                should_move,
                lambda: new_state.cars[lane, 0] + dir,
                lambda: new_state.cars[lane, 0],
            )
            
            # Wrap around screen
            new_x = jnp.where(
                self._env.consts.car_update[lane] > 0,
                jnp.where(
                    new_x > self._env.consts.screen_width,
                    -self._env.consts.car_width,
                    new_x
                ),
                jnp.where(
                    new_x < -self._env.consts.car_width,
                    self._env.consts.screen_width,
                    new_x
                ),
            ).astype(jnp.int32)
            
            fast_cars = fast_cars.at[lane, 0].set(new_x)
        
        return new_state._replace(cars=fast_cars)


class BlackCarsMod(JaxAtariInternalModPlugin):
    """Makes all cars black by overriding CAR_COLORS constant"""
    
    # Override CAR_COLORS to make all cars black
    # Using (0, 0, 0) for pure black. None means use original color.
    constants_overrides = {
        "CAR_COLORS": [
            (0, 0, 0),  # Lane 0 - black
            (0, 0, 0),  # Lane 1 - black
            (0, 0, 0),  # Lane 2 - black
            (0, 0, 0),  # Lane 3 - black
            (0, 0, 0),  # Lane 4 - black
            (0, 0, 0),  # Lane 5 - black
            (0, 0, 0),  # Lane 6 - black
            (0, 0, 0),  # Lane 7 - black
            (0, 0, 0),  # Lane 8 - black
            (0, 0, 0),  # Lane 9 - black
        ]
    }
