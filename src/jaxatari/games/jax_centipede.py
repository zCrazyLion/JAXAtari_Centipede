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
#from jaxatari.rendering.jax_rendering_utils import recolor_sprite


class CentipedeConstants:
    # -------- Game constants --------
    WIDTH = 160
    HEIGHT = 210
    SCALING_FACTOR = 6

    ## -------- Player constants --------
    PLAYER_START_X = 76
    PLAYER_START_Y = 172
    PLAYER_BOUNDS = (16, 140), (141, 172)

    PLAYER_SIZE = (4, 9)

    MAX_VELOCITY_X = 1 # Default: 6 | Maximum speed in x direction (pixels per frame)
    ACCELERATION_X = 1 # Default: 0.2 | How fast player accelerates
    FRICTION_X = 1 # Default: 1 | 1 = 100% -> player stops immediately, 0 = 0% -> player does not stop, 0.5 = 50 % -> player loses 50% of its velocity every frame
    PLAYER_Y_VALUES = jnp.array([141, 145, 145.5, 147, 147.5, 150, 150.5, 154, 154.5, 156, 156.5, 159, 159.5, 163, 163.5, 165, 165.5, 168, 168.5, 172])      # Double to not need extra state value
    MAX_VELOCITY_Y = 0.25

    ## -------- Player spell constants --------
    PLAYER_SPELL_SPEED = 4.5

    PLAYER_SPELL_SIZE = (0, 8) # (0, 8) because of collision logic of spell

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

    ## -------- Spider constants --------
    SPIDER_X_POSITIONS = jnp.array([16, 133])
    SPIDER_Y_POSITIONS = jnp.array([115, 124, 133, 142, 151, 160, 169, 178])
    SPIDER_MOVE_PROBABILITY = 0.2
    SPIDER_MIN_SPAWN_FRAMES = 55
    SPIDER_MAX_SPAWN_FRAMES = 355
    SPIDER_SIZE = (8, 6)
    SPIDER_CLOSE_RANGE = 16
    SPIDER_MID_RANGE = 32

    ## -------- Color constants --------
    ORANGE = jnp.array([181, 83, 40])#B55328    # Mushrooms lvl1
    DARK_ORANGE = jnp.array([198, 108, 58])#C66C3A
    PINK = jnp.array([184, 70, 162])#B846A2      # Centipede lvl1
    GREEN = jnp.array([110, 156, 66])#6E9C42     # Border lvl1
    LIGHT_PURPLE = jnp.array([188, 144, 252])#BC90FC  # UI Elements
    PURPLE = jnp.array([146, 70, 192])#9246C0        # Spider
    DARK_PURPLE = jnp.array([66, 72, 200])#4248C8
    LIGHT_BLUE = jnp.array([84, 138, 210])#548AD2    # Border lvl2
    DARK_BLUE = jnp.array([45, 50, 184])#2D32B8     # Mushrooms lvl2
    LIGHT_RED = jnp.array([200, 72, 72])#C84848
    RED = jnp.array([184, 50, 50])#B83232       # Centipede lvl2
    YELLOW = jnp.array([187, 187, 53])#BBBB35
    DARK_YELLOW = jnp.array([162, 162, 42])#A2A22A
    # POISONED_RAINBOW = ?

    ## -------- Sprite Frames --------
    SPRITE_PLAYER_FRAMES = 1
    SPRITE_PLAYER_SPELL_FRAMES = 1
    SPRITE_CENTIPEDE_FRAMES = 1
    SPRITE_MUSHROOM_FRAMES = 1
    SPRITE_SPIDER_FRAMES = 4
    SPRITE_SPIDER_300_FRAMES = 1
    SPRITE_SPIDER_600_FRAMES = 1
    SPRITE_SPIDER_900_FRAMES = 1
    SPRITE_FLEA_FRAMES = 2
    SPRITE_SCORPION_FRAMES = 2
    SPRITE_SPARKS_FRAMES = 4
    SPRITE_BOTTOM_BORDER_FRAMES = 1

    # -------- Centipede States --------

class CentipedeState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_velocity_x: chex.Array
    player_spell: chex.Array  # (1, 3): x, y, is_alive
    mushroom_positions: chex.Array # (304, 4): x, y, is_poisoned, lives; 304 mushrooms in total
    centipede_position: chex.Array # (9, 5): x, y, speed(horizontal), movement(vertical), status/is_head; 9 segments in total
    spider_position: chex.Array # (1, 3): x, y, direction
    spider_spawn_timer: chex.Array # Frames until new spider spawns
    spider_points: chex.Array # (1, 2): sprite, timeout
    # flea_position: chex.Array # (1, 3) array for flea: (x, y, lives), 2 lives, speed doubles after 1 hit
    # scorpion_position: chex.Array # (1, 3) array for scorpion: (x, y, direction)
    score: chex.Array
    lives: chex.Array
    wave: chex.Array # (1, 2): logical wave, ui wave
    step_counter: chex.Array
    rng_key: chex.PRNGKey

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
    spider_300 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider_scores/300.npy"))
    spider_600 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider_scores/600.npy"))
    spider_900 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/spider_scores/900.npy"))
    flea1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/flea/1.npy"))
    flea2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/flea/2.npy"))
    scorpion1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/scorpion/1.npy"))
    scorpion2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/scorpion/2.npy"))
    sparks1 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/1.npy"))
    sparks2 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/2.npy"))
    sparks3 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/3.npy"))
    sparks4 = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/sparks/4.npy"))
    bottom_border = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/bottom_border.npy"))

    sparks_sprites, _ = jru.pad_to_match([sparks1, sparks2, sparks3, sparks4])
    spider_sprites, _ = jru.pad_to_match([spider1, spider2, spider3, spider4])
    flea_sprites, _ = jru.pad_to_match([flea1, flea2])
    scorpion_sprites, _ = jru.pad_to_match([scorpion1, scorpion2])

    SPRITE_PLAYER = jnp.expand_dims(player, 0)
    SPRITE_PLAYER_SPELL = jnp.expand_dims(player_spell, 0)

    SPRITE_CENTIPEDE = jnp.expand_dims(centipede, 0)
    SPRITE_MUSHROOM = jnp.expand_dims(mushroom, 0)

    SPRITE_SPIDER = jnp.concatenate(
        [
            jnp.repeat(spider_sprites[0][None], 8, axis=0),
            jnp.repeat(spider_sprites[1][None], 8, axis=0),
            jnp.repeat(spider_sprites[2][None], 8, axis=0),
            jnp.repeat(spider_sprites[3][None], 8, axis=0),
        ]
    )
    SPRITE_SPIDER_300 = jnp.expand_dims(spider_300, 0)
    SPRITE_SPIDER_600 = jnp.expand_dims(spider_600, 0)
    SPRITE_SPIDER_900 = jnp.expand_dims(spider_900, 0)
    SPRITE_FLEA = jnp.concatenate(
        [
            jnp.repeat(flea_sprites[0][None], 2, axis=0),
            jnp.repeat(flea_sprites[1][None], 2, axis=0),
        ]
    )
    SPRITE_SCORPION = jnp.concatenate(
        [
            jnp.repeat(scorpion_sprites[0][None], 8, axis=0),
            jnp.repeat(scorpion_sprites[1][None], 8, axis=0),
        ]
    )

    SPRITE_SPARKS = jnp.concatenate(
        [
            jnp.repeat(sparks_sprites[0][None], 4, axis=0),
            jnp.repeat(sparks_sprites[1][None], 4, axis=0),
            jnp.repeat(sparks_sprites[2][None], 4, axis=0),
            jnp.repeat(sparks_sprites[3][None], 4, axis=0),
        ]
    )
    SPRITE_BOTTOM_BORDER = jnp.expand_dims(bottom_border, 0)


    DIGITS = jru.load_and_pad_digits(os.path.join(MODULE_DIR, "sprites/centipede/big_numbers/{}.npy"))
    LIFE_INDICATOR = jru.loadFrame(os.path.join(MODULE_DIR, "sprites/centipede/ui/wand.npy"))

    return (
        SPRITE_PLAYER,
        SPRITE_PLAYER_SPELL,
        SPRITE_CENTIPEDE,
        SPRITE_MUSHROOM,
        SPRITE_SPIDER,
        SPRITE_SPIDER_300,
        SPRITE_SPIDER_600,
        SPRITE_SPIDER_900,
        SPRITE_FLEA,
        SPRITE_SCORPION,
        SPRITE_SPARKS,
        SPRITE_BOTTOM_BORDER,
        DIGITS,
        LIFE_INDICATOR,
    )

(
    SPRITE_PLAYER,
    SPRITE_PLAYER_SPELL,
    SPRITE_CENTIPEDE,
    SPRITE_MUSHROOM,
    SPRITE_SPIDER,
    SPRITE_SPIDER_300,
    SPRITE_SPIDER_600,
    SPRITE_SPIDER_900,
    SPRITE_FLEA,
    SPRITE_SCORPION,
    SPRITE_SPARKS,
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
        self.renderer = CentipedeRenderer(self.consts)

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
        """Returns the observation space for Centipede.
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
            # TODO: fill
        })

    def image_space(self) -> spaces.Box:
        """Returns the image space for Centipede.
        The image is an RGB image with shape (160, 210, 3).
        """
        return spaces.Box(
            low=0,
            high=255,
            shape=(160, 210, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0, ))
    def _get_observation(self, state: CentipedeState) -> CentipedeObservation:      # TODO: fill
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
    def _get_info(self, state: CentipedeState, all_rewards: jnp.ndarray) -> CentipedeInfo:      # TODO: fill
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

    # -------- Helper Functions --------

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
    def get_mushroom_index(self, pos: chex.Array) -> chex.Array:
        row_idx = (pos[1] - 7) / 9
        odd_row = pos[1] % 2 == 0
        col_idx = jnp.where(odd_row, pos[0], pos[0] - 4) / 8
        return row_idx * 16 + col_idx - 2

    # -------- Logic Functions --------

    ## -------- Centipede Move Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def centipede_step(
        self,
        centipede_state: chex.Array,
        mushrooms_positions: chex.Array,
    ) -> chex.Array:

        def should_turn_around(i, carry):
            groups, same_group = carry
            this_seg = centipede_state[i]
            next_seg = centipede_state[i + 1]

            is_head = jnp.where(this_seg[4] == 2, True, False)

            def head_case():
                def same_group_head():
                    return groups.at[i].set(1), True

                def dif_group():
                    return groups, False

                dist = jnp.linalg.norm(this_seg[:2] - next_seg[:2])
                return jax.lax.cond(jnp.less_equal(dist, 10), same_group_head, dif_group)

            def same_group_tail():
                return groups, True

            return jax.lax.cond(is_head, head_case, same_group_tail)

        init_carry = jnp.zeros_like(centipede_state[:, 0], dtype=jnp.int32), False
        turn_around, _ = jax.lax.fori_loop(0, centipede_state.shape[0] - 1, should_turn_around, init_carry)

        def move_segment(segment, turn_around):
            moving_left = segment[2] < 0

            def step_horizontal():
                speed = segment[2] * 0.25    # should be 0.21875 but see yourself
                new_x = segment[0] + speed
                return jnp.array([new_x, segment[1],segment[2], segment[3], segment[4]]), jnp.array(0)

            def step_vertical():
                moving_down = jnp.greater(segment[3], 0)
                y_dif = jnp.array(jnp.where(moving_down, self.consts.MUSHROOM_Y_SPACING, -self.consts.MUSHROOM_Y_SPACING))
                new_y = segment[1] + y_dif
                new_horizontal_direction = jnp.where(
                    jnp.logical_or(jnp.greater_equal(new_y, 176), jnp.logical_and(jnp.less(segment[3], 0), jnp.less_equal(new_y, 131))),
                    -segment[3],
                    segment[3]
                )
                new_status = jnp.where(segment[4] == 1.5, 2, segment[4])    # Change to head if moved vertically

                return jnp.array([
                    segment[0],
                    new_y,
                    -segment[2],
                    new_horizontal_direction,
                    new_status,
                ]), jnp.array(0)

            def step_turn_around():
                new_speed = -segment[2]
                speed = new_speed * 0.25    # should be 0.21875 but see yourself
                new_x = segment[0] + speed
                return jnp.array([new_x, segment[1], new_speed, segment[3], segment[4]]), jnp.array(1)

            def check_mushroom_collision(mushroom, seg_speed):
                direction = jnp.where(seg_speed > 0, 6, -6)
                collision = jnp.logical_and(
                    self.check_collision_single(
                        pos1=jnp.array([segment[0], segment[1]]),
                        size1=self.consts.SEGMENT_SIZE,
                        pos2=mushroom[:2],
                        size2=self.consts.MUSHROOM_SIZE
                    ),
                    self.check_collision_single(
                        pos1=jnp.array([segment[0] + direction, segment[1]]),
                        size1=self.consts.SEGMENT_SIZE,
                        pos2=mushroom[:2],
                        size2=self.consts.MUSHROOM_SIZE
                    ),
                )

                return jnp.where(mushroom[3] > 0, collision, False)

            collision = jax.vmap(lambda m: check_mushroom_collision(m, segment[2]))(mushrooms_positions)
            collision = jnp.any(collision)

            move_down = jnp.logical_or(
                collision,
                jnp.logical_or(
                    jnp.logical_and(jnp.less_equal(segment[0], self.consts.PLAYER_BOUNDS[0][0]), moving_left),
                    jnp.logical_and(jnp.greater_equal(segment[0], self.consts.PLAYER_BOUNDS[0][1]), jnp.invert(moving_left))
                )
            )
            return jax.lax.cond(
                move_down,
                lambda: jax.lax.cond(jnp.logical_and(jnp.greater_equal(segment[1], 176), turn_around == 1),
                                     step_turn_around,
                                     step_vertical),
                step_horizontal)

        new_state, segment_split = jax.vmap(move_segment)(centipede_state, turn_around)

        segment_split = jnp.roll(segment_split, 1)
        def set_new_status(seg, split):     # change value of split following segment so this can be set as head later
            return jnp.where(split == 1, seg.at[4].set(1.5), seg)

        return jax.vmap(set_new_status)(new_state, segment_split)

    ## -------- Spell Mushroom Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0, ))
    def check_spell_mushroom_collision(
        self,
        spell_state: chex.Array,
        mushroom_positions: chex.Array,
        score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        def check_single_mushroom(is_alive, mushroom, score):
            def no_hit():
                return is_alive, mushroom, score

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
                    new_score = jnp.where(new_hp == 0, score + 1, score)
                    return False, updated_mushroom_position, new_score

                def check_hp():
                    return jax.lax.cond(mush_hp > 0, on_hit, lambda: (is_alive, mushroom, score))

                return jax.lax.cond(collision, check_hp, lambda: (is_alive, mushroom, score))

            return jax.lax.cond(is_alive != 0, check_hit, no_hit)

        spell_active, updated_mushrooms, updated_score = jax.vmap(
            lambda m: check_single_mushroom(spell_is_alive, m, score)
        )(mushroom_positions)

        spell_active = jnp.invert(jnp.any(jnp.invert(spell_active)))
        return spell_state.at[2].set(jnp.where(spell_active, 1, 0)), updated_mushrooms, jnp.max(updated_score)

    ## -------- Spider Mushroom Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spider_mushroom_collision(
            self,
            spider_position: chex.Array,
            mushroom_positions: chex.Array,  # shape (304, 4): x, y, is_poisoned, lives
    ) -> chex.Array:
        """
        Detects collisions between spider and mushrooms.
        Returns updated mushroom positions (lives set to 0 if hit by spider).
        """

        spider_x, spider_y, spider_dir = spider_position
        spider_alive = spider_dir != 0

        def no_collision():
            return mushroom_positions.astype(jnp.float32)

        def check_hit():
            # Prüfe pro Mushroom Kollision mit der Spinne
            def collide_single(mushroom):
                x, y, is_poisoned, lives = mushroom

                collision = self.check_collision_single(
                    pos1=jnp.array([spider_x + 4, spider_y - 2]),
                    size1=(2, 4), # mushrooms do not always react to the spider bumping into them so the spider frame to check has to be smaller than SPIDER_SIZE (so we take MUSHROOM_SIZE so that two mushrooms cannot be hit at the same time)
                    pos2=jnp.array([x, y]),
                    size2=self.consts.MUSHROOM_SIZE,
                )

                # Bei Kollision direkt lives auf 0 setzen
                new_lives = jnp.where(collision, 0.0, lives)

                return jnp.array([x, y, is_poisoned, new_lives], dtype=jnp.float32)

            new_mushrooms = jax.vmap(collide_single)(mushroom_positions)
            return new_mushrooms

        return jax.lax.cond(spider_alive, check_hit, no_collision)

    ## -------- Spider Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_spell_spider_collision(
            self,
            spell_state: chex.Array,
            spider_position: chex.Array,
            score: chex.Array,
            player_y: chex.Array,
            spider_points: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """
        Detects collision between spell and spider.
        Points are determined by the distance from player to spider on hit
        - close range: 900 Points
        - middle range: 600 Points
        - far range: 300 Points
        Additionally returns a spider_timer_sprite as int32[2] array.
        """

        # Check if spell is still active
        spell_pos_x = spell_state[0]
        spell_pos_y = spell_state[1]
        spell_is_alive = spell_state[2] != 0

        # Check if spider is still active
        spider_x, spider_y, spider_dir = spider_position
        spider_alive = spider_dir != 0

        # Default return (no collision, no sprite)
        def no_collision():
            return spell_state, spider_position, score, spider_points

        def check_hit():
            collision = self.check_collision_single(
                pos1=jnp.array([spell_pos_x, spell_pos_y]),
                size1=self.consts.PLAYER_SPELL_SIZE,
                pos2=jnp.array([spider_x + 2, spider_y - 2]),
                size2=self.consts.SPIDER_SIZE,
            )

            def on_hit():
                # Distance from player to spider
                dist = jnp.abs(player_y - spider_y)

                # Points determined by distance
                points = jnp.where(
                    dist < self.consts.SPIDER_CLOSE_RANGE,
                    900,
                    jnp.where(
                        dist < self.consts.SPIDER_MID_RANGE,
                        600,
                        300,
                    ),
                )

                # Sprite determined by distance
                spider_timer_sprite = jnp.select(
                    [
                        dist < self.consts.SPIDER_CLOSE_RANGE,
                        dist < self.consts.SPIDER_MID_RANGE,
                    ],
                    [
                        jnp.array([3, 55], dtype=jnp.int32),
                        jnp.array([2, 55], dtype=jnp.int32),
                    ],
                    default=jnp.array([1, 55], dtype=jnp.int32),
                )

                new_spell = spell_state.at[2].set(0)
                new_spider = jnp.array([spider_x, spider_y, 0], dtype=jnp.float32)
                new_score = score + points
                return new_spell, new_spider, new_score, spider_timer_sprite

            return jax.lax.cond(collision, on_hit, no_collision)

        return jax.lax.cond(
            jnp.logical_and(spell_is_alive, spider_alive),
            check_hit,
            no_collision,
        )

    ## -------- Centipede Spell Collision Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def check_centipede_spell_collision(
            self,
            spell_state: chex.Array,
            centipede_position: chex.Array,
            mushroom_positions: chex.Array,
            score: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        spell_active = spell_state[2] != 0

        def no_hit():
            return (
                jnp.repeat(spell_active, centipede_position.shape[0]),
                centipede_position,
                jnp.repeat(0, centipede_position.shape[0]),
                jnp.repeat(0, centipede_position.shape[0]),
                jnp.repeat(-1, centipede_position.shape[0])
            )

        def check_single_segment(is_alive, seg):
            seg_pos = seg[:2]

            collision = self.check_collision_single(
                pos1=jnp.array([spell_state[0], spell_state[1]]),
                size1=self.consts.PLAYER_SPELL_SIZE,
                pos2=seg_pos,
                size2=self.consts.MUSHROOM_SIZE,
            )

            def on_hit():
                mush_y = seg[1] + 2
                odd_mush_row = seg[1] % 2 == 0
                mush_x = jnp.where(
                    odd_mush_row,
                    jnp.where(
                        seg[2] > 0,
                        jnp.ceil(seg[0] / 8) * 8,
                        jnp.floor(seg[0] / 8) * 8,
                    ),
                    jnp.where(
                        seg[2] > 0,
                        jnp.ceil(seg[0] / 8) * 8 + 4,
                        jnp.floor(seg[0] / 8) * 8 + 4,
                    )
                )
                out_of_border = jnp.where(
                    odd_mush_row,
                    jnp.logical_or(
                        jnp.logical_and(seg[2] > 0, mush_x > 136),
                        jnp.logical_and(seg[2] < 0, mush_x < 16)
                    ),
                    jnp.logical_or(
                        jnp.logical_and(seg[2] > 0, mush_x > 140),
                        jnp.logical_and(seg[2] < 0, mush_x < 20)
                    )
                )
                idx = jnp.where(out_of_border, -1, self.get_mushroom_index(jnp.array([mush_x, mush_y])))
                return (
                    False,
                    jnp.zeros_like(seg),
                    jnp.where(seg[4] == 2, 100, 10),
                    jnp.array(1),
                    jnp.array(idx, dtype=jnp.int32)
                )

            return jax.lax.cond(collision, on_hit, lambda: (is_alive, seg, 0, jnp.array(0), jnp.array(-1)))

        check = jax.vmap(lambda s: check_single_segment(spell_active, s), in_axes=0)

        (
            spell_active,
            new_centipede_position,
            new_score,
            segment_hit,
            mush_idx
        ) = jax.lax.cond(spell_active != 0, lambda: check(centipede_position), no_hit)
        spell_active = jnp.invert(jnp.any(jnp.invert(spell_active)))

        new_score = jnp.sum(new_score)
        new_heads = jnp.roll(segment_hit, 1)
        mush_idx = jnp.max(mush_idx)
        new_mushroom_positions = jnp.where(
            jnp.logical_and(
                jnp.logical_and(
                    mush_idx >= 0,
                    mush_idx < self.consts.MAX_MUSHROOMS
                ),
                mushroom_positions[mush_idx, 3] == 0
            ),
            mushroom_positions.at[mush_idx, 3].set(3),
            mushroom_positions
        )

        def set_new_status(seg, new):     # change value of hit following segment to head
            return jnp.where(jnp.logical_and(new == 1, seg[4] != 0), seg.at[4].set(2), seg)

        return (
            spell_state.at[2].set(jnp.where(spell_active, 1, 0)),
            jax.vmap(set_new_status)(new_centipede_position, new_heads),
            new_mushroom_positions,
            score + new_score
        )

    ## -------- Mushroom Spawn Logic -------- ##
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

    ## -------- Spider Spawn Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def initialize_spider_position(self, key: chex.PRNGKey) -> chex.Array:
        idx_x = jax.random.randint(key, (), 0, 2)
        x = self.consts.SPIDER_X_POSITIONS[idx_x]

        direction = jnp.where(x == 133, -1, +1)

        idx_y = jax.random.randint(key, (), 0, len(self.consts.SPIDER_Y_POSITIONS))
        y = self.consts.SPIDER_Y_POSITIONS[idx_y]

        return jnp.stack([x, y, direction]).astype(jnp.float32)

    ## -------- Centipede Spawn Logic -------- ##
    @partial(jax.jit, static_argnums=(0, ))
    def initialize_centipede_positions(self, wave: chex.Array) -> chex.Array:
        base_x = 80
        base_y = 5
        initial_positions = jnp.zeros((self.consts.MAX_SEGMENTS, 5))

        wave = wave[0]
        slow_wave = wave < 0
        num_heads = jnp.abs(wave)
        main_segments = self.consts.MAX_SEGMENTS - num_heads

        def spawn_segment(i, segments: chex.Array):
            def main_body():
                is_head = i == 0
                return segments.at[i].set(
                    jnp.where(
                        slow_wave,
                        jnp.where(
                            is_head,
                            jnp.array([base_x + 4 * i, base_y, -1, 1, 2]),
                            jnp.array([base_x + 4 * i, base_y, -1, 1, 1]),
                        ),
                        jnp.where(
                            is_head,
                            jnp.array([base_x + 4 * i, base_y, -2, 1, 2]),
                            jnp.array([base_x + 4 * i, base_y, -2, 1, 1]),
                        )
                    )
                )

            def single_head():      # May not be 100% accurate (1-2px offset, varying per round)
                j = i - main_segments
                return jnp.where(
                    j == 0,
                    segments.at[i].set(jnp.array([140, 5, -2, 1, 2])),      # TODO: sometimes starts with different direction
                    jnp.where(
                        j == 1,
                        segments.at[i].set(jnp.array([16, 5, 2, 1, 2])),
                        jnp.where(
                            j == 2,
                            segments.at[i].set(jnp.array([108, 5, -2, 1, 2])),
                            jnp.where(
                                j == 3,
                                segments.at[i].set(jnp.array([48, 14, 2, 1, 2])),
                                jnp.where(
                                    j == 4,
                                    segments.at[i].set(jnp.array([124, 23, 2, 1, 2])),
                                    jnp.where(
                                        j == 5,
                                        segments.at[i].set(jnp.array([32, 14, 2, 1, 2])),
                                        jnp.where(
                                            j == 6,
                                            segments.at[i].set(jnp.array([92, 14, -2, 1, 2])),
                                            jnp.where(
                                                j == 7,
                                                segments.at[i].set(jnp.array([64, 14, -2, 1, 2])),
                                                segments.at[i].set(jnp.array([80, 5, -2, 1, 2])),
                                            )
                                        )
                                    ),
                                )
                            )
                        )
                    )
                )

            return jax.lax.cond(
                i < main_segments,
                main_body,
                single_head,
            )

        return jax.lax.fori_loop(0, self.consts.MAX_SEGMENTS, spawn_segment, initial_positions)

    ## -------- Wave Logic -------- ##
    @partial(jax.jit, static_argnums=(0,))
    def process_wave(
            self,
            centipede_state: chex.Array,
            wave: chex.Array,
            score: chex.Array
    ) -> tuple[chex.Array, chex.Array]:
        logical_wave = wave[0]
        ui_wave = wave[1]

        def new_wave():
            new_logical_wave = jnp.where(
                score < 40_000,
                jnp.where(
                    logical_wave < 0,
                    logical_wave * -1 % 8,
                    (logical_wave + 1) * -1 % -8
                ),
                jnp.abs(logical_wave) + 1 % 8
            )

            new_wave = jnp.array([new_logical_wave, ui_wave + 1 % 8])
            return self.initialize_centipede_positions(new_wave), new_wave

        return jax.lax.cond(
            jnp.sum(centipede_state[:, 4]) == 0,
            lambda: new_wave(),
            lambda: (centipede_state, wave)
        )

    ## -------- Player Move Logic -------- ##
    @partial(jax.jit, static_argnums=(0, ))
    def player_step(
            self,
            player_x: chex.Array,
            player_y: chex.Array,
            player_velocity_x: chex.Array,
            action: chex.Array
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
        acc_dir = jnp.where(right, 1, jnp.where(left, -1, 0))
        no_horiz_op = jnp.invert(jnp.logical_or(right, left))
        turn_around = jnp.logical_or(
            jnp.logical_and(jnp.greater(player_velocity_x, 1/32), left),
            jnp.logical_and(jnp.less(player_velocity_x, -1/32), right),
        )

        raw_vel_x = jnp.fix(player_velocity_x) * 0.5
        new_velocity_x = jnp.where(
            no_horiz_op,
            jnp.where(
                player_x % 4 == 0,
                0,
                1/32 * jnp.sign(player_velocity_x),
            ),
            jnp.where(
                turn_around,
                1/32 * jnp.sign(player_velocity_x),
                jnp.clip(
                    jnp.where(jnp.abs(raw_vel_x) * 2 < 1, player_velocity_x + 1/8 * acc_dir, player_velocity_x + 1/16 * acc_dir),
                    -3, 3,
                ),
            )
        )
        new_player_x = jnp.where(
            jnp.logical_and(no_horiz_op, player_x % 4 != 0),
            player_x + jnp.sign(new_velocity_x) * 0.5,
            jnp.clip(player_x + raw_vel_x, self.consts.PLAYER_BOUNDS[0][0], self.consts.PLAYER_BOUNDS[0][1])
        )

        # Calculate new y position
        y_idx = jnp.argmax(self.consts.PLAYER_Y_VALUES == player_y)
        new_idx = jnp.clip(
            y_idx + jnp.where(up, -1, jnp.where(down, 1, 0)),
            0, self.consts.PLAYER_Y_VALUES.shape[0] - 1)
        new_player_y = self.consts.PLAYER_Y_VALUES[new_idx]

        return new_player_x, new_player_y, new_velocity_x

    ## -------- Spider Move Logic -------- ##
    def spider_step(
            self,
            spider_x: chex.Array,
            spider_y: chex.Array,
            spider_direction: chex.Array,
            step_counter: chex.Array,
            key: chex.PRNGKey
    ) -> chex.Array:
        """Moves Spider one Step further with random x-movement and periodic y-movement"""

        # Split key in two parts, one for x and one for y
        key_x, key_y = jax.random.split(key)

        # X movement of spider
        move_x = jax.random.bernoulli(key_x, self.consts.SPIDER_MOVE_PROBABILITY)  # True = bewegen
        new_x = jnp.where(move_x, spider_x + spider_direction, spider_x)

        # Check if left or right border is reached
        stop_left = (spider_direction == -1) & (new_x < 16)
        stop_right = (spider_direction == +1) & (new_x > 133)

        new_direction = jnp.where(stop_left | stop_right, 0, spider_direction)

        # Y movement of spider
        move_y = (step_counter % 8 == 7)

        def update_y(spider_y):
            idx = jnp.argwhere(self.consts.SPIDER_Y_POSITIONS == spider_y, size=1).squeeze()

            can_go_up = idx > 0
            can_go_down = idx < len(self.consts.SPIDER_Y_POSITIONS) - 1

            rand = jax.random.bernoulli(key_y)

            dy = jnp.where(~can_go_down, -1,
                           jnp.where(~can_go_up, +1,
                                     jnp.where(rand, -1, +1)))

            new_idx = idx + dy
            return self.consts.SPIDER_Y_POSITIONS[new_idx].astype(jnp.float32)

        new_y = jax.lax.cond(
            move_y,
            update_y,
            lambda y: y.astype(jnp.float32),
            spider_y
        )

        return jnp.stack([new_x, new_y, new_direction])

    ## -------- Player Spell Logic -------- ##
    def player_spell_step(      # TODO: fix behaviour for close objects (add cooldown)
            self,
            player_x: chex.Array,
            player_y: chex.Array,
            player_spell: chex.Array,
            action: chex.Array
    ) -> chex.Array:

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

        spawn = jnp.logical_and(jnp.logical_not(player_spell[2] != 0), fire)

        new_is_alive = jnp.where(
            spawn,  # on spawn
            1,
            jnp.where(
                player_spell[1] < 0,
                0,
                player_spell[2]
            )  # on kill or keep
        )
        new_x = jnp.where(
            spawn,
            player_x + 1,
            jnp.where(new_is_alive, player_spell[0], 0.0)
        )
        new_y = jnp.where(
            spawn,
            jnp.floor(player_y) - 9,
            jnp.where(new_is_alive, player_spell[1] - self.consts.PLAYER_SPELL_SPEED, 0.0)
        )

        return jnp.array([new_x, new_y, new_is_alive])

    @partial(jax.jit, static_argnums=(0, ))
    def reset(self, key = 42) -> Tuple[CentipedeObservation, CentipedeState]:
        """Initialize game state"""

        key = jax.random.PRNGKey(time.time_ns() % (2 ** 32))  # Pseudo random number generator seed key, based on current system time.
        new_key0, key_spider = jax.random.split(key, 2)

        initial_spider_timer = jax.random.randint(key_spider, (), self.consts.SPIDER_MIN_SPAWN_FRAMES, self.consts.SPIDER_MAX_SPAWN_FRAMES + 1)

        reset_state = CentipedeState(
            player_x=jnp.array(self.consts.PLAYER_START_X),
            player_y=jnp.array(self.consts.PLAYER_START_Y),
            player_velocity_x=jnp.array(0),
            player_spell=jnp.zeros(3),
            mushroom_positions=self.initialize_mushroom_positions(),
            centipede_position=self.initialize_centipede_positions(jnp.array([0, 0])),
            spider_position=jnp.zeros(3),
            spider_spawn_timer=initial_spider_timer,
            spider_points=jnp.array([0, 0]),
            score=jnp.array(0),
            lives=jnp.array(3),
            step_counter=jnp.array(0),
            wave=jnp.array([0, 0]),
            rng_key=new_key0,
        )

        initial_obs = self._get_observation(reset_state)
        return initial_obs, reset_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: CentipedeState, action: chex.Array) -> tuple[
        CentipedeObservation, CentipedeState, float, bool, CentipedeInfo]:

        # --- Player Movement ---
        new_player_x, new_player_y, new_velocity_x = self.player_step(
            state.player_x, state.player_y, state.player_velocity_x, action
        )

        new_player_spell_state = self.player_spell_step(
            new_player_x, new_player_y, state.player_spell, action
        )

        # --- Mushroom Collision ---
        new_player_spell_state, updated_mushrooms, new_score = self.check_spell_mushroom_collision(
            new_player_spell_state, state.mushroom_positions, state.score
        )

        # --- Centipede Collision ---
        new_player_spell_state, new_centipede_position, updated_mushrooms, new_score = \
            self.check_centipede_spell_collision(
                new_player_spell_state, state.centipede_position, updated_mushrooms, new_score
            )

        # --- Spider collision with mushrooms ---
        updated_mushrooms = self.check_spider_mushroom_collision(
            state.spider_position,
            updated_mushrooms
        )

        # --- Centipede Step & Wave ---
        new_centipede_position = self.centipede_step(new_centipede_position, state.mushroom_positions)
        new_centipede_position, new_wave = self.process_wave(new_centipede_position, state.wave, state.score)

        # --- Spider Collision ---
        new_player_spell_state, new_spider_position, new_score, new_spider_points_pre = self.check_spell_spider_collision(
            new_player_spell_state,
            state.spider_position,
            new_score,
            new_player_y,
            state.spider_points,
        )

        # --- Spider Movement & Spawn Timer ---
        new_rng_key, key_spawn, key_step = jax.random.split(state.rng_key, 3)

        spider_x, spider_y, spider_dir = new_spider_position
        spider_alive = spider_dir != 0

        def move_spider(_):
            new_spider = self.spider_step(spider_x, spider_y, spider_dir, state.step_counter, key_step)
            return new_spider, state.spider_spawn_timer

        def handle_dead_spider(_):
            new_timer = state.spider_spawn_timer - 1

            def respawn():
                new_spider = self.initialize_spider_position(key_spawn).astype(jnp.float32)
                next_timer = jax.random.randint(
                    key_spawn,
                    (),
                    self.consts.SPIDER_MIN_SPAWN_FRAMES,
                    self.consts.SPIDER_MAX_SPAWN_FRAMES + 1
                )
                return new_spider, next_timer

            def wait():
                return jnp.array([state.spider_position[0], state.spider_position[1], 0], dtype=jnp.float32), new_timer

            return jax.lax.cond(new_timer <= 0, respawn, wait)

        new_spider_position, new_spider_timer = jax.lax.cond(
            spider_alive,
            move_spider,
            handle_dead_spider,
            operand=None
        )

        # --- Spider points ---
        new_spider_points = jnp.where(
            new_spider_points_pre[1] > 0,
            jnp.array([new_spider_points_pre[0], new_spider_points_pre[1] - 1]),
            new_spider_points_pre
        )

        # --- Return State ---
        return_state = state._replace(
            player_x=new_player_x,
            player_y=new_player_y,
            player_velocity_x=new_velocity_x,
            player_spell=new_player_spell_state,
            mushroom_positions=updated_mushrooms,
            centipede_position=new_centipede_position,
            spider_position=new_spider_position,
            spider_spawn_timer=new_spider_timer,
            spider_points=new_spider_points,
            score=new_score,
            step_counter=state.step_counter + 1,
            wave=new_wave,
            rng_key=new_rng_key
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

        def recolor_sprite(
                sprite: chex.Array,
                color: chex.Array,  # RGB, up to 4 dimensions
                bounds: tuple[int, int, int, int] = None  # (top, left, bottom, right)
        ) -> chex.Array:
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

        def get_sprite_frames(wave: chex.Array, step_counter: int):
            """Gives all sprite-frames dynamically depending on step_counter and recoloring depending on wave"""

            def get_frame(sprite_id, num_frames: int, step_counter: int):
                """Calculates dynamically the frame of a sprite"""
                if num_frames == 1:
                    return jru.get_sprite_frame(sprite_id, 0)
                idx = step_counter
                return jru.get_sprite_frame(sprite_id, idx)

            # --- Get frames dynamically --- #
            frame_player_idx = get_frame(SPRITE_PLAYER, self.consts.SPRITE_PLAYER_FRAMES, step_counter)
            frame_player_spell_idx = get_frame(SPRITE_PLAYER_SPELL, self.consts.SPRITE_PLAYER_SPELL_FRAMES, step_counter)
            frame_centipede_idx = get_frame(SPRITE_CENTIPEDE, self.consts.SPRITE_CENTIPEDE_FRAMES, step_counter)
            frame_mushroom_idx = get_frame(SPRITE_MUSHROOM, self.consts.SPRITE_MUSHROOM_FRAMES, step_counter)
            frame_spider_idx = get_frame(SPRITE_SPIDER, self.consts.SPRITE_SPIDER_FRAMES, step_counter)
            frame_spider300_idx = get_frame(SPRITE_SPIDER_300, self.consts.SPRITE_SPIDER_300_FRAMES, step_counter)
            frame_spider600_idx = get_frame(SPRITE_SPIDER_600, self.consts.SPRITE_SPIDER_600_FRAMES, step_counter)
            frame_spider900_idx = get_frame(SPRITE_SPIDER_900, self.consts.SPRITE_SPIDER_900_FRAMES, step_counter)
            frame_flea_idx = get_frame(SPRITE_FLEA, self.consts.SPRITE_FLEA_FRAMES, step_counter)
            frame_scorpion_idx = get_frame(SPRITE_SCORPION, self.consts.SPRITE_SCORPION_FRAMES, step_counter)
            frame_sparks_idx = get_frame(SPRITE_SPARKS, self.consts.SPRITE_SPARKS_FRAMES, step_counter)
            frame_bottom_border_idx = get_frame(SPRITE_BOTTOM_BORDER, self.consts.SPRITE_BOTTOM_BORDER_FRAMES, step_counter)

            # --- Recoloring depending on wave --- #
            recolored_sparks = recolor_sprite(frame_sparks_idx, self.consts.LIGHT_PURPLE)  # Platzhalter

            def wave_0():
                return (
                    recolor_sprite(frame_player_idx, self.consts.ORANGE),
                    recolor_sprite(frame_player_spell_idx, self.consts.ORANGE),
                    recolor_sprite(frame_centipede_idx, self.consts.PINK),
                    recolor_sprite(frame_mushroom_idx, self.consts.ORANGE),
                    recolor_sprite(frame_spider_idx, self.consts.PURPLE),
                    recolor_sprite(frame_spider300_idx, self.consts.PURPLE),
                    recolor_sprite(frame_spider600_idx, self.consts.GREEN),
                    recolor_sprite(frame_spider900_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_flea_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_scorpion_idx, self.consts.LIGHT_BLUE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.GREEN),
                )

            def wave_1():
                return (
                    recolor_sprite(frame_player_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_player_spell_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_centipede_idx, self.consts.RED),
                    recolor_sprite(frame_mushroom_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_spider_idx, self.consts.GREEN),
                    recolor_sprite(frame_spider300_idx, self.consts.GREEN),
                    recolor_sprite(frame_spider600_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_spider900_idx, self.consts.ORANGE),
                    recolor_sprite(frame_flea_idx, self.consts.YELLOW),
                    recolor_sprite(frame_scorpion_idx, self.consts.ORANGE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.LIGHT_BLUE),
                )

            def wave_2():
                return (
                    recolor_sprite(frame_player_idx, self.consts.YELLOW),
                    recolor_sprite(frame_player_spell_idx, self.consts.YELLOW),
                    recolor_sprite(frame_centipede_idx, self.consts.PURPLE),
                    recolor_sprite(frame_mushroom_idx, self.consts.YELLOW),
                    recolor_sprite(frame_spider_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_spider300_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_spider600_idx, self.consts.ORANGE),
                    recolor_sprite(frame_spider900_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_flea_idx, self.consts.PINK),
                    recolor_sprite(frame_scorpion_idx, self.consts.DARK_BLUE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.ORANGE),
                )

            def wave_3():
                return (
                    recolor_sprite(frame_player_idx, self.consts.PINK),
                    recolor_sprite(frame_player_spell_idx, self.consts.PINK),
                    recolor_sprite(frame_centipede_idx, self.consts.GREEN),
                    recolor_sprite(frame_mushroom_idx, self.consts.PINK),
                    recolor_sprite(frame_spider_idx, self.consts.ORANGE),
                    recolor_sprite(frame_spider300_idx, self.consts.ORANGE),
                    recolor_sprite(frame_spider600_idx, self.consts.DARK_PURPLE),
                    recolor_sprite(frame_spider900_idx, self.consts.YELLOW),
                    recolor_sprite(frame_flea_idx, self.consts.RED),
                    recolor_sprite(frame_scorpion_idx, self.consts.YELLOW),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.DARK_PURPLE),
                )

            def wave_4():
                return (
                    recolor_sprite(frame_player_idx, self.consts.RED),
                    recolor_sprite(frame_player_spell_idx, self.consts.RED),
                    recolor_sprite(frame_centipede_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_mushroom_idx, self.consts.RED),
                    recolor_sprite(frame_spider_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_spider300_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_spider600_idx, self.consts.DARK_YELLOW),
                    recolor_sprite(frame_spider900_idx, self.consts.PINK),
                    recolor_sprite(frame_flea_idx, self.consts.PURPLE),
                    recolor_sprite(frame_scorpion_idx, self.consts.PINK),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.DARK_YELLOW),
                )

            def wave_5():
                return (
                    recolor_sprite(frame_player_idx, self.consts.PURPLE),
                    recolor_sprite(frame_player_spell_idx, self.consts.PURPLE),
                    recolor_sprite(frame_centipede_idx, self.consts.ORANGE),
                    recolor_sprite(frame_mushroom_idx, self.consts.PURPLE),
                    recolor_sprite(frame_spider_idx, self.consts.YELLOW),
                    recolor_sprite(frame_spider300_idx, self.consts.YELLOW),
                    recolor_sprite(frame_spider600_idx, self.consts.PINK),
                    recolor_sprite(frame_spider900_idx, self.consts.RED),
                    recolor_sprite(frame_flea_idx, self.consts.GREEN),
                    recolor_sprite(frame_scorpion_idx, self.consts.RED),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.PINK),
                )

            def wave_6():
                return (
                    recolor_sprite(frame_player_idx, self.consts.GREEN),
                    recolor_sprite(frame_player_spell_idx, self.consts.GREEN),
                    recolor_sprite(frame_centipede_idx, self.consts.DARK_BLUE),
                    recolor_sprite(frame_mushroom_idx, self.consts.GREEN),
                    recolor_sprite(frame_spider_idx, self.consts.PINK),
                    recolor_sprite(frame_spider300_idx, self.consts.PINK),
                    recolor_sprite(frame_spider600_idx, self.consts.RED),
                    recolor_sprite(frame_spider900_idx, self.consts.PURPLE),
                    recolor_sprite(frame_flea_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_scorpion_idx, self.consts.PURPLE),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.RED),
                )

            def wave_7():
                return (
                    recolor_sprite(frame_player_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_player_spell_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_centipede_idx, self.consts.YELLOW),
                    recolor_sprite(frame_mushroom_idx, self.consts.LIGHT_BLUE),
                    recolor_sprite(frame_spider_idx, self.consts.RED),
                    recolor_sprite(frame_spider300_idx, self.consts.RED),
                    recolor_sprite(frame_spider600_idx, self.consts.RED),
                    recolor_sprite(frame_spider900_idx, self.consts.GREEN),
                    recolor_sprite(frame_flea_idx, self.consts.ORANGE),
                    recolor_sprite(frame_scorpion_idx, self.consts.GREEN),
                    recolored_sparks,
                    recolor_sprite(frame_bottom_border_idx, self.consts.RED),
                )

            wave_mod = wave[1] % 8

            return jax.lax.switch(
                wave_mod,
                [wave_0, wave_1, wave_2, wave_3, wave_4, wave_5, wave_6, wave_7],
            )

        (
            frame_player,
            frame_player_spell,
            frame_centipede,
            frame_mushroom,
            frame_spider,
            frame_spider300,
            frame_spider600,
            frame_spider900,
            frame_flea,
            frame_scorpion,
            frame_sparks,
            frame_bottom_border,
        ) = get_sprite_frames(state.wave, state.step_counter)

        ### -------- Render mushrooms -------- ###
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

        ### -------- Render spider -------- ###
        raster = jnp.where(
            state.spider_position[2] != 0,
            jru.render_at(
                raster,
                state.spider_position[0] + 2,
                state.spider_position[1] - 2,
                frame_spider,
            ),
            raster
        )

        ### -------- Render spider score -------- ###
        raster = jnp.where(
            state.spider_points[1] != 0,
            jru.render_at(
                raster,
                state.spider_position[0] + 2,
                state.spider_position[1] - 2,
                jnp.where(
                    state.spider_points[0] == 1,
                    frame_spider300,
                    jnp.where(
                        state.spider_points[0] == 2,
                        frame_spider600,
                        frame_spider900,
                    )
                )
            ),
            raster
        )

        ### -------- Render centipede -------- ###
        def render_centipede_segment(i, raster_base):
            should_render = state.centipede_position[i][4] != 0
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

        ### -------- Render player -------- ###
        raster = jru.render_at(
            raster,
            state.player_x,
            state.player_y,
            frame_player,
        )

        ### -------- Render player spell -------- ###
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

        ### -------- Render bottom border -------- ###
        raster = jru.render_at(
            raster,
            16,
            183,
            frame_bottom_border,
        )

        ### -------- Render score -------- ###
        score_array = jru.int_to_digits(state.score, max_digits=6)
        # first_nonzero = jnp.argmax(score_array != 0)
        # _, score_array = jnp.split(score_array, first_nonzero - 1)
        raster = jru.render_label(raster, 100, 187, score_array, DIGITS, spacing=8)

        ### -------- Render live indicator -------- ###
        raster = jru.render_indicator(raster, 16, 187, state.lives - 1, LIFE_INDICATOR, spacing=8)

        return raster