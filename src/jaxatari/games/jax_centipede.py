"""

Lukas Bergholz, Linus Orlob, Vincent Jahn

"""

import os
import jax
import jax.numpy as jnp
import chex
import pygame
import jaxatari.rendering.jax_rendering_utils as jru
import time
from functools import partial
from typing import NamedTuple, Tuple

from jaxatari import spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action, EnvState, EnvObs
from jaxatari.renderers import JAXGameRenderer

class CentipedeConstants:
    # -------- Game constants --------
    WIDTH = 160
    HEIGHT = 210
    SCALING_FACTOR = 6

    ## -------- Player constants --------
    PLAYER_START_X = 76
    PLAYER_START_Y = 190 - 18
    PLAYER_BOUNDS = (16, 140), (150, 172) # TODO: Check if correct

    PLAYER_SIZE = (4, 9)

    MAX_VELOCITY_X = 6 # Default: 6 | Maximum speed in x direction (pixels per frame)
    ACCELERATION_X = 0.2 # Default: 0.2 | How fast player accelerates
    FRICTION_X = 1 # Default: 1 | 1 = 100% -> player stops immediately, 0 = 0% -> player does not stop, 0.5 = 50 % -> player loses 50% of its velocity every frame
    MAX_VELOCITY_Y = 2.5 # Default: 2.5 | Maximum speed in y direction (pixels per frame)

    ## -------- Player spell constants --------
    PLAYER_SPELL_SPEED = 10

    PLAYER_SPELL_SIZE = (0, 8) # (0, 8) because of collision logic from spell

    ## -------- Starting Pattern (X -> placed, O -> not placed) --------
    MUSHROOM_STARTING_PATTERN = [
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

    ## -------- Mushroom constants --------
    MAX_MUSHROOMS = 304             # Default 304 (19*16) | Maximum number of mushrooms that can appear at the same time
    MUSHROOM_NUMBER_OF_ROWS = 19    # Default 19 | Number of rows -> Determines value of MAX_MUSHROOMS
    MUSHROOM_NUMBER_OF_COLS = 16    # Default 16 | Number of mushrooms per row -> Determines value of MAX_MUSHROOMS
    MUSHROOM_X_SPACING = 8      #
    MUSHROOM_Y_SPACING = 9
    MUSHROOM_COLUMN_START_EVEN = 20
    MUSHROOM_COLUMN_START_ODD = 16
    MUSHROOM_SIZE = (4, 3)
    MUSHROOM_HITBOX_Y_OFFSET = 6

    ## -------- Centipede constants --------
    MAX_SEGMENTS = 9
    SEGMENT_SIZE = (4, 6)
"""
# -------- States -------- 
class PlayerSpellState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    is_alive: jnp.ndarray
    """

class CentipedeState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_velocity_x: chex.Array
    player_spell: chex.Array  # (1, 3) array for player spell: x, y, is_alive
    mushroom_positions: chex.Array # (304, 5) array for mushroom positions -> mushroom_positions need 5 entries per mushroom: 1. x value 2. y value 3. is shown 4. lives (1, 2 or 3) 5. is poisoned -> there are also 304 mushrooms in total
    centipede_position: chex.Array # (9, 4) must contain position, direction and status(head)
    # spider_position: chex.Array # (1, 3) array for spider (x, y, direction)
    # flea_position: chex.Array # (1, 3) array for flea, 2 lives, speed doubles after 1 hit
    # scorpion_position: chex.Array # (1, ?) array for scorpion, only moves from right to left?: (x, y)
    score: chex.Array
    lives: chex.Array
    wave: chex.Array # number of wave (+ 0.5 if second (faster) wave
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

    player = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/player/player.npy"))
    player_spell = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/player_spell/player_spell.npy"))
    mushroom = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/mushrooms/mushroom.npy"))
    centipede = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/centipede/segment.npy"))
    spider1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/1.npy"))
    spider2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/2.npy"))
    spider3 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/3.npy"))
    spider4 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider/4.npy"))
    flea1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/flea/1.npy"))
    bottom_border = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/bottom_border.npy"))

    spider_sprites, _ = jru.pad_to_match([spider1, spider2, spider3, spider4])

    SPRITE_PLAYER = jnp.expand_dims(player, 0)
    SPRITE_PLAYER_spell = jnp.expand_dims(player_spell, 0)

    SPRITE_CENTIPEDE = jnp.expand_dims(centipede, 0)
    SPRITE_FLEA = jnp.expand_dims(flea1, 0)
    SPRITE_SPIDER = jnp.concatenate(
        [
            jnp.repeat(spider_sprites[0][None], 4, axis=0),
            jnp.repeat(spider_sprites[1][None], 4, axis=0),
            jnp.repeat(spider_sprites[2][None], 4, axis=0),
            jnp.repeat(spider_sprites[3][None], 4, axis=0),
        ]
    )

    #jax.debug.print("{}", centipede.shape)

    SPRITE_BOTTOM_BORDER = jnp.expand_dims(bottom_border, 0)
    SPRITE_MUSHROOM = jnp.expand_dims(mushroom, 0)

    DIGITS = jru.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/centipede/big_numbers/{}.npy"))
    LIFE_INDICATOR = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/wand.npy"))

    # Debug
    frame_player = jru.get_sprite_frame(SPRITE_PLAYER, 0)
    #jax.debug.print("{x}, {y}", x=frame_player, y=(frame_player.shape[0], frame_player.shape[1], frame_player.shape[2]))

    return (
        SPRITE_PLAYER,
        SPRITE_PLAYER_spell,
        SPRITE_MUSHROOM,
        SPRITE_CENTIPEDE,
        SPRITE_SPIDER,
        SPRITE_FLEA,
        SPRITE_BOTTOM_BORDER,
        DIGITS,
        LIFE_INDICATOR,
    )

(
    SPRITE_PLAYER,
    SPRITE_PLAYER_spell,
    SPRITE_MUSHROOM,
    SPRITE_CENTIPEDE,
    SPRITE_SPIDER,
    SPRITE_FLEA,
    SPRITE_BOTTOM_BORDER,
    DIGITS,
    LIFE_INDICATOR,
) = load_sprites()

# -------- Game Logic --------

class JaxCentipede(JaxEnvironment[CentipedeState, CentipedeObservation, CentipedeInfo, CentipedeConstants]):
    def __init__(self, consts: CentipedeConstants = None, reward_funcs: list[callable] =None):
        consts = consts or CentipedeConstants()
        super().__init__(consts)
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
        player = PlayerEntity(
            x=state.player_x,
            y=state.player_y,
            o=state.player_velocity_x,
            width=jnp.array(self.consts.PLAYER_SIZE[0]),
            height=jnp.array(self.consts.PLAYER_SIZE[1]),
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
    def check_collision_single(self, pos1, size1, pos2, size2):
        """Check collision between two single entities"""
        # Calculate edges for rectangle 1
        rect1_left = pos1[0]
        rect1_right = pos1[0] + size1[0]
        rect1_top = pos1[1]
        rect1_bottom = pos1[1] + size1[1]

        # Calculate edges for rectangle 2
        rect2_left = pos2[0]
        rect2_right = pos2[0] + size2[0]
        rect2_top = pos2[1]
        rect2_bottom = pos2[1] + size2[1]

        # Check overlap
        horizontal_overlap = jnp.logical_and(
            rect1_left < rect2_right,
            rect1_right > rect2_left
        )

        vertical_overlap = jnp.logical_and(
            rect1_top < rect2_bottom,
            rect1_bottom > rect2_top
        )

        return jnp.logical_and(horizontal_overlap, vertical_overlap)

    @partial(jax.jit, static_argnums=(0, ))
    def check_spell_collision_with_mushrooms(
        self,
        spell_pos_x: chex.Array,
        spell_pos_y: chex.Array,
        spell_is_alive: chex.Array,
        mushroom_positions: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        def check_single_mushroom(is_alive, mushroom):

            def no_hit():
                return is_alive, mushroom

            def check_hit():
                mush_pos = mushroom[:2]
                mush_hp = mushroom[3]

                collision = self.check_collision_single(
                    pos1=jnp.array([spell_pos_x, spell_pos_y + self.consts.MUSHROOM_HITBOX_Y_OFFSET]),
                    size1=self.consts.PLAYER_SPELL_SIZE,
                    pos2=mush_pos,
                    size2=self.consts.MUSHROOM_SIZE
                )

                def on_hit():
                    new_hp = mush_hp - 1
                    updated_mushroom_position = mushroom.at[3].set(new_hp)
                    return False, updated_mushroom_position

                def check_hp():
                    return jax.lax.cond(mush_hp > 0, on_hit, lambda: (is_alive, mushroom))

                return jax.lax.cond(collision, check_hp, lambda: (is_alive, mushroom))

            return jax.lax.cond(is_alive != 0, check_hit, no_hit)

        spell_active, updated_mushrooms = jax.vmap(
            lambda m: check_single_mushroom(spell_is_alive, m)
        )(mushroom_positions)

        spell_active = jnp.invert(jnp.any(jnp.invert(spell_active)))
        return jnp.where(spell_active, 1, 0), updated_mushrooms

    ## -------- Mushroom Spawn Logic --------
    @partial(jax.jit, static_argnums=(0, ))
    def initialize_mushroom_positions(self) -> chex.Array:
        # Create row and column indices
        row_indices = jnp.repeat(jnp.arange(self.consts.MUSHROOM_NUMBER_OF_ROWS), self.consts.MUSHROOM_NUMBER_OF_COLS)
        col_indices = jnp.tile(jnp.arange(self.consts.MUSHROOM_NUMBER_OF_COLS), self.consts.MUSHROOM_NUMBER_OF_ROWS)

        # Compute row parity
        row_is_even = row_indices % 2 == 0
        column_start = jnp.where(row_is_even, self.consts.MUSHROOM_COLUMN_START_EVEN, self.consts.MUSHROOM_COLUMN_START_ODD)
        x = column_start + self.consts.MUSHROOM_X_SPACING * col_indices
        x = x.astype(jnp.int32)

        y = (row_indices * self.consts.MUSHROOM_Y_SPACING + 7).astype(jnp.int32)

        # Build full pattern as array
        pattern_array = jnp.array([
            [1 if c.upper() == 'X' else 0 for c in row.ljust(self.consts.MUSHROOM_NUMBER_OF_COLS, 'O')]
            for row in self.consts.MUSHROOM_STARTING_PATTERN
        ])
        pattern_array = jnp.pad(
            pattern_array,
            ((0, max(0, self.consts.MUSHROOM_NUMBER_OF_ROWS - pattern_array.shape[0])), (0, 0)),
            constant_values=0
        )

        lives = pattern_array[row_indices, col_indices] * 3  # 3 lives if visible, 0 if not
        is_poisoned = jnp.zeros_like(lives)

        return jnp.stack([x, y, is_poisoned, lives], axis=1)

    @partial(jax.jit, static_argnums=(0, ))
    def spawn_centipede(self, wave: jnp.ndarray) -> chex.Array:
        base_x = 79
        base_y = 5
        initial_positions = jnp.zeros((self.consts.MAX_SEGMENTS, 4))

        def spawn_segment(i, segments: jnp.ndarray):
            is_head = i == 0
            return segments.at[i].set(
                jnp.where(
                    is_head,
                    jnp.array([base_x + 4*i, base_y, 1, 11]),
                    jnp.array([base_x + 4*i, base_y, 1, 1]),
                )
            )

        centipede = jax.lax.fori_loop(0, self.consts.MAX_SEGMENTS, spawn_segment, initial_positions)
        jax.debug.print("{}", centipede)
        return centipede

    @partial(jax.jit, static_argnums=(0, ))
    def player_step(
            self, state: CentipedeState, action: chex.Array
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
        accel_x = jnp.where(right, self.consts.ACCELERATION_X, jnp.where(left, -self.consts.ACCELERATION_X, 0.0))  # Compute acceleration based on input

        # x velocity
        velocity_x = state.player_velocity_x  # Get current x velocity

        moving_left_right_input = jnp.logical_and(right, (velocity_x < 0)) # If currently moving to the left and right input is detected
        moving_right_left_input = jnp.logical_and(left, (velocity_x > 0)) # If currently moving to the right and left input is detected

        direction_change = jnp.where(jnp.logical_or(moving_left_right_input, moving_right_left_input),True,False) # Detect direction change and reset velocity if needed
        velocity_x = jnp.where(direction_change, 0.0, velocity_x)  # Reset velocity on direction change
        velocity_x = velocity_x + accel_x  # Update velocity with acceleration
        velocity_x = jnp.where(jnp.logical_not(jnp.logical_or(left, right)), velocity_x * (1.0 - self.consts.FRICTION_X), velocity_x)  # Slow down if no input
        velocity_x = jnp.clip(velocity_x, -self.consts.MAX_VELOCITY_X, self.consts.MAX_VELOCITY_X)  # Clamp velocity within limits

        # Global x position
        new_player_x = state.player_x + velocity_x  # Compute next x position
        velocity_x = jnp.where(new_player_x <= self.consts.PLAYER_BOUNDS[0][0], 0.0, velocity_x)  # Stop at left bound
        player_x = jnp.clip(state.player_x + velocity_x, self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[0][1])  # Final x position

        # Calculate new y position
        delta_y = jnp.where(up, -self.consts.MAX_VELOCITY_Y, jnp.where(down, self.consts.MAX_VELOCITY_Y, 0))
        player_y = jnp.clip(state.player_y + delta_y, self.consts.PLAYER_BOUNDS[1][0], self.consts.PLAYER_BOUNDS[1][1])

        return player_x, player_y, velocity_x


    def player_spell_step(
            self, state: CentipedeState, action: chex.Array
    ) -> jnp.array:

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
        kill_spell = jnp.logical_and(state.player_spell[1] < 0, collision_with_mushrooms)
        spawn = jnp.logical_and(jnp.logical_not(state.player_spell[2] != 0), fire)

        # New is_alive-status
        new_is_alive = jnp.where(
            spawn,  # on spawn
            True,
            jnp.where(kill_spell, False, state.player_spell[2] != 0)  # on kill or keep
        )
        '''
        for i in range(0, 304, 50):
            jax.debug.print("Mushroom positions {i}–{j}: {chunk}",
                            i=i,
                            j=i + 50,
                            chunk=state.mushroom_positions[i:i + 50])
        '''
        # Base x
        base_x = jnp.where(spawn, state.player_x + 1, state.player_spell[0]) # player x on spawn or keep x
        # Base y
        base_y = jnp.where(spawn, state.player_y + 5, state.player_spell[1]) # player y on spawn or keey y

        # move only when alive
        new_y = jnp.where(
            spawn,
            state.player_y + 5 - self.consts.PLAYER_SPELL_SPEED,
            jnp.where(new_is_alive, base_y - self.consts.PLAYER_SPELL_SPEED, 0.0)
        )
        new_x = jnp.where(
            new_is_alive,
            base_x,
            0.0
        )

        return jnp.array([new_x, new_y, new_is_alive])

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(time.time_ns() % (2**32))) -> tuple[CentipedeObservation, CentipedeState]:
        """Initialize game state"""
        reset_state = CentipedeState( # TODO: fill
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_velocity_x=jnp.array(0),
            player_spell=jnp.zeros(3),
            mushroom_positions=self.initialize_mushroom_positions(),
            centipede_position=self.spawn_centipede(wave=jnp.array(0)),
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

        new_player_x, new_player_y, new_velocity_x = self.player_step(state, action)

        new_player_spell_state = self.player_spell_step(state, action)

        spell_active, updated_mushrooms = self.check_spell_collision_with_mushrooms(
            new_player_spell_state[0],
            new_player_spell_state[1],
            new_player_spell_state[2] != 0,
            state.mushroom_positions,
        )

        new_player_spell_state = new_player_spell_state.at[2].set(spell_active) #_replace(is_alive=spell_active)

        return_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_velocity_x=new_velocity_x,
            player_spell=new_player_spell_state,
            mushroom_positions=updated_mushrooms,
            step_counter=state.step_counter + 1
        )

        obs = self._get_observation(return_state)
        all_rewards = self._get_all_rewards(state, return_state)
        info = self._get_info(return_state, all_rewards)

        return obs, return_state, 0.0, False, info

class CentipedeRenderer(JAXGameRenderer):
    def __init__(self, consts: CentipedeConstants = None):
        super().__init__()
        self.consts = consts or CentipedeConstants()

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: CentipedeState):
        raster = jnp.zeros((self.consts.HEIGHT, self.consts.WIDTH, 3))

        def recolor_sprite(  # TODO: recolor sprites only when colors change (new wave)
                sprite: jnp.ndarray,
                color: jnp.ndarray,  # RGB, up to 4 dimensions
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
        frame_mushroom = jru.get_sprite_frame(SPRITE_MUSHROOM, 0)
        frame_mushroom = recolor_sprite(frame_mushroom, jnp.array([92, 186, 92]))

        def render_mushrooms(i, raster_base):
            should_render = state.mushroom_positions[i][3] > 0
            return jax.lax.cond(
                should_render,
                lambda r: jru.render_at(
                    r,
                    state.mushroom_positions[i][0],
                    state.mushroom_positions[i][1],
                    frame_mushroom,
                ),
                lambda r: r,
                raster_base,
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_MUSHROOMS, render_mushrooms, raster)

        ### -------- Render centipede --------
        frame_centipede = jru.get_sprite_frame(SPRITE_CENTIPEDE, 0)
        frame_centipede = recolor_sprite(frame_centipede, jnp.array([92, 186, 92]))

        def render_centipede_segment(i, raster_base):
            should_render = state.centipede_position[i][2] != 0
            return jax.lax.cond(
                should_render,
                lambda r: jru.render_at(
                    r,
                    state.centipede_position[i][0],
                    state.centipede_position[i][1],
                    frame_centipede,
                ),
                lambda r: r,
                raster_base
            )

        raster = jax.lax.fori_loop(0, self.consts.MAX_SEGMENTS, render_centipede_segment, raster)

        ### -------- Render player --------
        frame_player = jru.get_sprite_frame(SPRITE_PLAYER, 0)
        frame_player = recolor_sprite(frame_player, jnp.array([92, 186, 92]))
        raster = jru.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_player,
        )

        ### -------- Render player spell --------
        frame_player_spell = jru.get_sprite_frame(SPRITE_PLAYER_spell, 0)
        frame_player_spell = recolor_sprite(frame_player_spell, jnp.array([92, 186, 92]))
        raster = jnp.where(
            state.player_spell[2] != 0,
            jru.render_at(
                raster,
                state.player_spell[0],
                state.player_spell[1],
                frame_player_spell,
            ),
        raster
        )

        ### -------- Render bottom border --------
        frame_bottom_border = jru.get_sprite_frame(SPRITE_BOTTOM_BORDER, 0)
        frame_bottom_border = recolor_sprite(frame_bottom_border, jnp.array([92, 186, 92]))
        raster = jru.render_at(
            raster,
            16,
            183,
            frame_bottom_border,
        )

        ### -------- Render score -------- TODO: make colorable, dynamic digit count
        score_array = jru.int_to_digits(state.score, max_digits=6)
        # first_nonzero = jnp.argmax(score_array != 0)
        # _, score_array = jnp.split(score_array, first_nonzero - 1)
        raster = jru.render_label(raster, 100, 187, score_array, DIGITS, spacing=8)

        ### -------- Render live indicator --------
        life_indicator = recolor_sprite(LIFE_INDICATOR, jnp.array([92, 186, 92]))
        raster = jru.render_indicator(raster, 16, 187, state.lives - 1, life_indicator, spacing=8)

        return raster

"""

def get_human_action() -> chex.Array:
    Get human action from keyboard with support for diagonal movement and combined fire
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
    
"""