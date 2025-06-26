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

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvState, EnvObs
from jaxatari.games.jax_seaquest import SeaquestObservation, LIFE_INDICATOR
from jaxatari.renderers import AtraJaxisRenderer
from jaxatari.rendering.atraJaxis import render_indicator

# -------- Game constants --------
WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 4

## -------- Starting Pattern (X -> placed, O -> not placed) --------
CENTIPEDE_STARTING_PATTERN = [
        "OOOOOOOOOOOOOOOO",
        "OOOOOOOOXOOOOXOO",
        "OOOOOOOOOXOOOOXO",
        "OOOOOOOOOOXXOOOO",
        "OOOXOOOOOOOOOOOO",
        "OXOOOOOOOOOOOOOO",
        "OOOOOOOXOOOOOOOO",
        "OOOOOOXOOOOOOOXO",
        "OOOXXOOOOXOOOOOO",
        "OOOOOOOXOOOOOOOO",
        "OOOOOOOOOOOOXOOO",
        "OOOOXOOOOOOOOOOO",
        "OOOOOOOOOOOOOXOO",
        "OOOOOOOOOOOXOOOX",
        "OOOOOOOOOOOOOOXO",
        "OOOOXOOXOOOOOOOO",
        "OXOOOXOOOOOOOOOO",
        "OOOOXOOOOOOOOOOX",
        "OOOOOOOOOOOOOOOO",
    ]

## -------- Player constants --------
PLAYER_START_X = 76
PLAYER_START_Y = 190 - 18
PLAYER_BOUNDS = (16, 140), (150, 172) # TODO: Check if correct

PLAYER_SIZE = (4, 9)

MAX_VELOCITY_X = 6 # Default: 6 | Maximum speed in x direction (pixels per frame)
ACCELERATION_X = 0.2 # Default: 0.2 | How fast player accelerates
FRICTION_X = 1 # Default: 1 | 1 = 100% -> player stops immediately, 0 = 0% -> player does not stop, 0.5 = 50 % -> player loses 50% of its velocity every frame
MAX_VELOCITY_Y = 2.5 # Default: 2.5 | Maximum speed in y direction (pixels per frame)

## -------- Player missile constants --------
PLAYER_MISSILE_SPEED = 10

PLAYER_MISSILE_SIZE = (1, 5) # TODO: Sprite may be (1, 6), right now it is (1, 5)

## -------- Mushroom constants --------
MAX_MUSHROOMS = 304             # Default 304 (19*16) | Maximum number of mushrooms that can appear at the same time
MUSHROOM_NUMBER_OF_ROWS = 19    # Default 19 | Number of rows -> Determines value of MAX_MUSHROOMS
MUSHROOM_NUMBER_OF_COLS = 16    # Default 16 | Number of mushrooms per row -> Determines value of MAX_MUSHROOMS
MUSHROOM_X_SPACING = 8      #
MUSHROOM_Y_SPACING = 9
MUSHROOM_COLUMN_START_EVEN = 20
MUSHROOM_COLUMN_START_ODD = 16

# -------- States --------
class PlayerMissileState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    is_alive: jnp.ndarray

class CentipedeState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_velocity_x: chex.Array
    player_missile: PlayerMissileState
    mushroom_positions: chex.Array # (304, 5) array for mushroom positions -> mushroom_positions need 5 entries per mushroom: 1. x value 2. y value 3. is shown 4. lives (1, 2 or 3) 5. is poisoned -> there are also 304 mushrooms in total
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

class PlayerEntity(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    o: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray
    active: jnp.ndarray

class CentipedeObservation(NamedTuple):
    player: PlayerEntity
    # mushrooms: jnp.ndarray
    # centipede: jnp.ndarray
    # spider: jnp.ndarray
    # flea: jnp.ndarray
    # scorpion: jnp.ndarray
    # score: jnp.ndarray
    # lives: jnp.ndarray
    # TODO: fill
    # if changed: obs_to_flat_array, _get_observation, (step, reset)

class CentipedeInfo(NamedTuple):
    # difficulty: jnp.ndarray # add if necessary
    step_counter: jnp.ndarray
    all_rewards: jnp.ndarray

# -------- Render Constants --------
def load_sprites():
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    player = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/player/player.npy"))
    player_missile = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/player_missile/player_missile.npy"))
    flea = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/flea/1.npy"))
    spider = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/1.npy"))
    bottom_border = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/bottom_border.npy"))
    mushroom = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/mushrooms/mushroom.npy"))

    SPRITE_PLAYER = jnp.expand_dims(player, 0)
    SPRITE_PLAYER_MISSILE = jnp.expand_dims(player_missile, 0)
    SPRITE_FLEA = jnp.expand_dims(flea, 0)
    SPRITE_SPIDER = jnp.expand_dims(spider, 0)
    SPRITE_BOTTOM_BORDER = jnp.expand_dims(bottom_border, 0)
    SPRITE_MUSHROOM = jnp.expand_dims(mushroom, 0)

    DIGITS = aj.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/centipede/big_numbers/{}.npy"))
    LIFE_INDICATOR = aj.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/wand.npy"))

    # Debug
    frame_player = aj.get_sprite_frame(SPRITE_PLAYER, 0)
    #jax.debug.print("{x}, {y}", x=frame_player, y=(frame_player.shape[0], frame_player.shape[1], frame_player.shape[2]))

    jax.debug.print("{}", SPRITE_SPIDER[:,:,:,0])

    return (
        SPRITE_PLAYER,
        SPRITE_PLAYER_MISSILE,
        SPRITE_FLEA,
        SPRITE_SPIDER,
        SPRITE_BOTTOM_BORDER,
        SPRITE_MUSHROOM,
        DIGITS,
        LIFE_INDICATOR,
    )

(
    SPRITE_PLAYER,
    SPRITE_PLAYER_MISSILE,
    SPRITE_FLEA,
    SPRITE_SPIDER,
    SPRITE_BOTTOM_BORDER,
    SPRITE_MUSHROOM,
    DIGITS,
    LIFE_INDICATOR,
) = load_sprites()

# -------- Game Logic --------

## -------- Mushroom Spawn Logic --------
def initialize_mushroom_positions() -> chex.Array: # TODO: make jittable

    mushroom_positions = []

    for row in range(MUSHROOM_NUMBER_OF_ROWS):
        row_is_even = row % 2 == 0
        column_counter = MUSHROOM_COLUMN_START_EVEN if row_is_even else MUSHROOM_COLUMN_START_ODD
        row_y = row * MUSHROOM_Y_SPACING + 7

        for col in range(MUSHROOM_NUMBER_OF_COLS):
            x = int(column_counter)
            y = int(row_y)

            # Sichtbarkeit anhand des Patterns
            char = CENTIPEDE_STARTING_PATTERN[row][col].upper() if row < len(CENTIPEDE_STARTING_PATTERN) and col < len(CENTIPEDE_STARTING_PATTERN[row]) else "O"
            is_shown = 1 if char == "X" else 0

            is_poisoned = 0
            lives = 3

            mushroom_positions.append([x, y, is_shown, is_poisoned, lives])
            column_counter += MUSHROOM_X_SPACING

    return jnp.array(mushroom_positions)

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


def player_missile_step(
        state: CentipedeState, action: chex.Array
) -> PlayerMissileState:

    fire = jnp.isin(action, jnp.array([
        Action.FIRE,
        Action.UPFIRE,
        Action.RIGHTFIRE,
        Action.LEFTFIRE,
        Action.DOWNFIRE,
        Action.UPRIGHTFIRE,
        Action.UPLEFTFIRE,
        Action.DOWNRIGHTFIRE,
        Action.DOWNLEFTFIRE
    ]))

    collision_with_mushrooms = True # TODO: Implement
    kill_missile = jnp.logical_and(state.player_missile.y < 0, collision_with_mushrooms)
    spawn = jnp.logical_and(jnp.logical_not(state.player_missile.is_alive), fire)

    # New is_alive-status
    new_is_alive = jnp.where(
        spawn,  # on spawn
        True,
        jnp.where(kill_missile, False, state.player_missile.is_alive)  # on kill or keep
    )

    # Base x
    base_x = jnp.where(spawn, state.player_x + 1, state.player_missile.x) # player x on spawn or keep x
    # Base y
    base_y = jnp.where(spawn, state.player_y + 5, state.player_missile.y) # player y on spawn or keey y

    # move only when alive
    new_y = jnp.where(
        spawn,
        state.player_y + 5 - PLAYER_MISSILE_SPEED,
        jnp.where(new_is_alive, base_y - PLAYER_MISSILE_SPEED, 0.0)
    )
    new_x = jnp.where(
        new_is_alive,
        base_x,
        0.0
    )

    return PlayerMissileState(x=new_x, y=new_y, is_alive=new_is_alive)


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

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CentipedeState) -> jnp.ndarray:
        """Render the game state to a raster image."""
        return self.renderer.render(state)

    def flatten_entity_position(self, entity: EntityPosition) -> jnp.ndarray:
        return jnp.concatenate([jnp.array([entity.x]), jnp.array([entity.y]), jnp.array([entity.width]), jnp.array([entity.height]), jnp.array([entity.active])])

    def flatten_player_entity(self, entity: PlayerEntity) -> jnp.ndarray:
        return jnp.concatenate([jnp.array([entity.x]), jnp.array([entity.y]), jnp.array([entity.o]), jnp.array([entity.width]), jnp.array([entity.height]), jnp.array([entity.active])])

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: CentipedeObservation) -> jnp.ndarray:
        return jnp.concatenate([
            self.flatten_player_entity(obs.player)
            # TODO: fill
        ])

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Seaquest.
                The observation contains:
        """
        return spaces.Dict({
            "player": spaces.Dict({
                "x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "o": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
                "width": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "height": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "active": spaces.Box(low=0, high=1, shape=(), dtype=jnp.int32),
            }),
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Seaquest.
        The image is a RGB image with shape (160, 210, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(160, 210, 3),
            dtype=jnp.uint8
        )

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
            player_missile=PlayerMissileState(x=jnp.array(0), y=jnp.array(0), is_alive=jnp.array(False)),
            mushroom_positions=initialize_mushroom_positions(),
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

        new_player_missile_state = player_missile_step(state, action)

        return_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_velocity_x=new_velocity_x,
            player_missile=new_player_missile_state,
            step_counter=state.step_counter + 1
        )

        obs = self._get_observation(return_state)
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)

        return obs, return_state, 0.0, False, info

class CentipedeRenderer(AtraJaxisRenderer):
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CentipedeState):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        def recolor_sprite( # TODO: recolor sprites only when colors change (new wave)
                sprite: jnp.ndarray,
                color: jnp.ndarray, # RGB, up to 4 dimensions
                bounds: tuple[int, int, int, int] = None  # (top, left, bottom, right)
        ) -> jnp.ndarray:
            # Ensure color is the same dtype as sprite
            dtype = sprite.dtype
            color = color.astype(dtype)

            assert sprite.ndim == 3 and sprite.shape[2] in (3, 4), "Sprite must be HxWx3 or HxWx4"

            if color.shape[0] < sprite.shape[2]:
                missing = sprite.shape[2] - color.shape[0]
                pad = jnp.full((missing,), 255, dtype=dtype)
                color = jnp.concatenate([color, pad], axis=0)


            assert color.shape[0] == sprite.shape[2], "Color channels must match sprite channels"

            H, W, _ = sprite.shape

            if bounds is None:
                region = sprite
            else:
                top, left, bottom, right = bounds
                assert 0 <= left < right <= H and 0 <= top < bottom <= W, "Invalid bounds"
                region = sprite[left:right, top:bottom]

            visible_mask = jnp.any(region != 0, axis=-1, keepdims=True)  # (h, w, 1)

            color_broadcasted = jnp.broadcast_to(color, region.shape).astype(dtype)
            recolored_region = jnp.where(visible_mask, color_broadcasted, jnp.zeros_like(color_broadcasted))

            if bounds is None:
                return recolored_region
            else:
                recolored_sprite = sprite.at[left:right, top:bottom].set(recolored_region)
                return recolored_sprite

        ### -------- Render mushrooms --------
        frame_mushroom = aj.get_sprite_frame(SPRITE_MUSHROOM, state.step_counter)
        frame_mushroom = recolor_sprite(frame_mushroom, jnp.array([92, 186, 92]))

        def render_mushrooms(i, raster_base):
            should_render = state.mushroom_positions[i][2] == 1
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    state.mushroom_positions[i][0],
                    state.mushroom_positions[i][1],
                    frame_mushroom,
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, MAX_MUSHROOMS, render_mushrooms, raster)

        ### -------- Render player --------
        frame_player = aj.get_sprite_frame(SPRITE_PLAYER, 0)
        frame_player = recolor_sprite(frame_player, jnp.array([92, 186, 92]))
        raster = aj.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_player,
        )

        ### -------- Render player missile --------
        frame_player_missile = aj.get_sprite_frame(SPRITE_PLAYER_MISSILE, 0)
        frame_player_missile = recolor_sprite(frame_player_missile, jnp.array([92, 186, 92]))
        raster = jnp.where(
            state.player_missile.is_alive,
            aj.render_at(
                raster,
                state.player_missile.x,
                state.player_missile.y,
                frame_player_missile,
            ),
        raster
        )

        ### -------- Render bottom border --------
        frame_bottom_border = aj.get_sprite_frame(SPRITE_BOTTOM_BORDER, 0)
        frame_bottom_border = recolor_sprite(frame_bottom_border, jnp.array([92, 186, 92]))
        raster = aj.render_at(
            raster,
            16,
            183,
            frame_bottom_border,
        )

        ### -------- Render score -------- TODO: make colorable, dynamic digit count
        score_array = aj.int_to_digits(state.score, max_digits=5)
        # first_nonzero = jnp.argmax(score_array != 0)
        # _, score_array = jnp.split(score_array, first_nonzero - 1)
        raster = aj.render_label(raster, 108, 187, score_array, DIGITS, spacing=8)

        ### -------- Render live indicator --------
        life_indicator = recolor_sprite(LIFE_INDICATOR, jnp.array([92, 186, 92]))
        raster = render_indicator(raster, 16, 187, state.lives - 1, life_indicator, spacing=8)

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