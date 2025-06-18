"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""

import os
import jax
import jax.numpy as jnp
import chex
import pygame
import jaxatari.rendering.atraJaxis as aj
import time
from functools import partial
from typing import NamedTuple
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvState, EnvObs
from jaxatari.renderers import AtraJaxisRenderer

# -------- Game constants --------
WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 4

# -------- Player constants --------
PLAYER_START_X = 78
PLAYER_START_Y = 190
PLAYER_BOUNDS = (10, 146), (150, 180) # TODO: Check if correct

PLAYER_SIZE = (4, 9)

MAX_VELOCITY_X = 6 # Default: 6 | Maximum speed in x direction (pixels per frame)
ACCELERATION_X = 0.2 # Default: 0.2 | How fast player accelerates
FRICTION_X = 1 # Default: 1 | 1 = 100% -> player stops immediately, 0 = 0% -> player does not stop, 0.5 = 50 % -> player loses 50% of its velocity every frame
MAX_VELOCITY_Y = 2.5 # Default: 2.5 | Maximum speed in y direction (pixels per frame)

# -------- States --------
class CentipedeState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_velocity_x: chex.Array
    # mushroom_positions: chex.Array # (128, 2) array for mushroom positions
    # centipede_position: chex.Array # (9, ?) must contain position, direction, speed and if head
    # spider_position: chex.Array # (1, 3) array for spider (x, y, direction)
    # flea_position: chex.Array # (1, 3) array for flea, 2 lives, speed doubles after 1 hit
    # scorpion_position: chex.Array # (1, ?) array for scorpion, only moves from right to left?: (x, y)
    score: chex.Array
    lives: chex.Array
    wave: chex.Array
    step_counter: chex.Array
    rng_key: jax.random.PRNGKey
    # TODO: fill

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class CentipedeObservation(NamedTuple):
    player: EntityPosition
    # mushrooms: jnp.ndarray
    # centipede: jnp.ndarray
    # spider: jnp.ndarray
    # flea: jnp.ndarray
    # scorpion: jnp.ndarray
    # TODO: fill

class CentipedeInfo(NamedTuple):
    # difficulty: jnp.ndarray # add if necessary
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray

# -------- Render Constants --------
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/player/player.npy"))
    flea = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/enemies/flea.npy"))
    spider = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/enemies/spider.npy"))

    SPRITE_PLAYER = jnp.expand_dims(player, 0)
    SPRITE_FLEA = jnp.expand_dims(flea, 0)
    SPRITE_SPIDER = jnp.expand_dims(spider, 0)

    # Debug
    frame_player = aj.get_sprite_frame(SPRITE_PLAYER, 0)
    #jax.debug.print("{x}, {y}", x=frame_player, y=(frame_player.shape[0], frame_player.shape[1], frame_player.shape[2]))

    return (
        SPRITE_PLAYER,
        SPRITE_FLEA,
        SPRITE_SPIDER,
    )

(
    SPRITE_PLAYER,
    SPRITE_FLEA,
    SPRITE_SPIDER,
) = load_sprites()

# -------- Game Logic --------


@jax.jit
def player_step(
        state: CentipedeState, action: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array]:
    up = jnp.isin(action, jnp.array([
        Action.UP,
        Action.UPRIGHT,
        Action.UPLEFT,
        Action.UPFIRE,
        Action.UPRIGHTFIRE,
        Action.UPLEFTFIRE
    ]))
    down = jnp.isin(action, jnp.array([
        Action.DOWN,
        Action.DOWNRIGHT,
        Action.DOWNLEFT,
        Action.DOWNFIRE,
        Action.DOWNRIGHTFIRE,
        Action.DOWNLEFTFIRE
    ]))
    left = jnp.isin(action, jnp.array([
        Action.LEFT,
        Action.UPLEFT,
        Action.DOWNLEFT,
        Action.LEFTFIRE,
        Action.UPLEFTFIRE,
        Action.DOWNLEFTFIRE
    ]))
    right = jnp.isin(action, jnp.array([
        Action.RIGHT,
        Action.UPRIGHT,
        Action.DOWNRIGHT,
        Action.RIGHTFIRE,
        Action.UPRIGHTFIRE,
        Action.DOWNRIGHTFIRE
    ]))

    # x acceleration
    accel_x = jnp.where(right, ACCELERATION_X, jnp.where(left, -ACCELERATION_X, 0.0))  # Compute acceleration based on input

    # x velocity
    velocity_x = state.player_velocity_x  # Get current x velocity

    moving_left_right_input = jnp.logical_and(right, (velocity_x < 0)) # If currently moving to the left and right input is detected
    moving_right_left_input = jnp.logical_and(left, (velocity_x > 0)) # If currently moving to the right and left input is detected

    direction_change = jnp.where(jnp.logical_or(moving_left_right_input, moving_right_left_input),True,False) # Detect direction change and reset velocity if needed
    velocity_x = jnp.where(direction_change, 0.0, velocity_x)  # Reset velocity on direction change
    velocity_x = velocity_x + accel_x  # Update velocity with acceleration
    velocity_x = jnp.where(jnp.logical_not(jnp.logical_or(left, right)), velocity_x * (1.0 - FRICTION_X), velocity_x)  # Slow down if no input
    velocity_x = jnp.clip(velocity_x, -MAX_VELOCITY_X, MAX_VELOCITY_X)  # Clamp velocity within limits

    # Global x position
    new_player_x = state.player_x + velocity_x  # Compute next x position
    velocity_x = jnp.where(new_player_x <= PLAYER_BOUNDS[0][0], 0.0, velocity_x)  # Stop at left bound
    player_x = jnp.clip(state.player_x + velocity_x, PLAYER_BOUNDS[0][0], PLAYER_BOUNDS[0][1])  # Final x position

    # Calculate new y position
    delta_y = jnp.where(up, -MAX_VELOCITY_Y, jnp.where(down, MAX_VELOCITY_Y, 0))
    player_y = jnp.clip(state.player_y + delta_y, PLAYER_BOUNDS[1][0], PLAYER_BOUNDS[1][1])

    return player_x, player_y, velocity_x

class JaxCentipede(JaxEnvironment[CentipedeState, CentipedeObservation, CentipedeInfo]):
    def __init__(self, reward_funcs: list[callable] =None):
        super().__init__()
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]
        # self.frame_stack_size = 4 # ???
        # self.obs_size = 1024 # ???

    # TODO: add other funtions if needed

    def get_action_space(self) -> jnp.ndarray:
        return jnp.array(self.action_set)

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: CentipedeState) -> CentipedeObservation:
        # TODO: fill
        player = EntityPosition(
            x=state.player_x,
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
            active=jnp.array(1),
        )

        def convert_to_entity(pos, size):
            return jnp.array([
                pos[0],  # x position
                pos[1],  # y position
                size[0],  # width
                size[1],  # height
                pos[2] != 0,  # active flag
            ])

        return CentipedeObservation(
            player=player,
        )

    @partial(jax.jit, static_argnums=(0, ))
    def _get_info(self, state: CentipedeState, all_rewards: jnp.ndarray) -> CentipedeInfo:
        # TODO: fill
        return CentipedeInfo(
            step_counter=state.step_counter,
            all_rewards=all_rewards,
        )

    @jax.jit
    def _get_env_reward(self, previous_state: CentipedeState, state: CentipedeState) -> jnp.ndarray:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(self, previous_state: CentipedeState, state: CentipedeState) -> jnp.ndarray:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array([reward_func(previous_state, state) for reward_func in self.reward_funcs])
        return rewards

    @jax.jit
    def _get_done(self, state: CentipedeState) -> bool:
        return state.lives < 0

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(time.time_ns() % (2**32))) -> tuple[CentipedeObservation, CentipedeState]:
        """Initialize game state"""
        reset_state = CentipedeState( # TODO: fill
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_velocity_x=jnp.array(0),
            score=jnp.array(0),
            lives=jnp.array(3),
            step_counter=jnp.array(0),
            wave=jnp.array(1),
            rng_key=key,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0, ))
    def step(
            self, state: CentipedeState, action: Action
    ) -> tuple[CentipedeObservation, CentipedeState, float, bool, CentipedeInfo]:
        # TODO: fill

        new_player_x, new_player_y, new_velocity_x = player_step(state, action)

        return_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_velocity_x=new_velocity_x,
            step_counter=state.step_counter + 1
        )

        obs = self._get_observation(return_state)
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)

        return obs, return_state, 0.0, False, info

class CentipedeRenderer(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        def recolor_sprite(sprite: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
            # if color.shape[0] == 3:
            #     jnp.append(color, jnp.array(255))
            assert sprite.ndim == 3 and sprite.shape[2] in (3, 4), "Sprite must be HxWx3 or HxWx4"
            assert color.shape[0] == sprite.shape[2], "Color channels must match sprite channels"

            # Define a visibility mask: pixel is visible if any of its channels > 0
            visible_mask = jnp.any(sprite != 0, axis=-1)  # (H, W)
            visible_mask = visible_mask[:, :, None]  # (H, W, 1) for broadcasting

            # Broadcast color to the same shape as sprite
            color_broadcasted = jnp.broadcast_to(color, sprite.shape)

            # Where visible, use the new color; otherwise keep black (zeros)
            return jnp.where(visible_mask, color_broadcasted, 0)

        frame_player = aj.get_sprite_frame(SPRITE_PLAYER, 0)
        frame_player = recolor_sprite(frame_player, jnp.array([92, 186, 92, 255]))
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_player,
        )

        return raster

def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(Action.UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(Action.UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(Action.DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(Action.DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(Action.UPFIRE)
    if down and fire:
        return jnp.array(Action.DOWNFIRE)
    if left and fire:
        return jnp.array(Action.LEFTFIRE)
    if right and fire:
        return jnp.array(Action.RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(Action.UPRIGHT)
    if up and left:
        return jnp.array(Action.UPLEFT)
    if down and right:
        return jnp.array(Action.DOWNRIGHT)
    if down and left:
        return jnp.array(Action.DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(Action.UP)
    if down:
        return jnp.array(Action.DOWN)
    if left:
        return jnp.array(Action.LEFT)
    if right:
        return jnp.array(Action.RIGHT)
    if fire:
        return jnp.array(Action.FIRE)

    return jnp.array(Action.NOOP)

if __name__ == "__main__":
    # Initialize game and renderer
    game = JaxCentipede()
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()

    renderer_AtraJaxis = CentipedeRenderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_obs, curr_state = jitted_reset()

    # Game loop with rendering
    running = True
    frame_by_frame = False
    frameskip = 1
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    frame_by_frame = not frame_by_frame
            elif event.type == pygame.KEYDOWN or (
                event.type == pygame.KEYUP and event.key == pygame.K_n
            ):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        curr_obs, curr_state, reward, done, info = jitted_step(
                            curr_state, action
                        )
                        print(f"Observations: {curr_obs}")
                        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_obs, curr_state, reward, done, info = jitted_step(
                    curr_state, action
                )

        # render and update pygame
        raster = renderer_AtraJaxis.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        counter += 1
        clock.tick(60)

    pygame.quit()