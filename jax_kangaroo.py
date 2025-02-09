from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
from jax import Array

from kangaroo_levels import (
    LevelConstants,
    Kangaroo_Level_1,
    Kangaroo_Level_2,
    Kangaroo_Level_3,
)

# -------- Action constants --------
NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = range(10)
(
    UPFIRE,
    RIGHTFIRE,
    LEFTFIRE,
    DOWNFIRE,
    UPRIGHTFIRE,
    UPLEFTFIRE,
    DOWNRIGHTFIRE,
    DOWNLEFTFIRE,
) = range(10, 18)
RESET = 18

# -------- Game constants --------
RENDER_SCALE_FACTOR = 4
SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210
PLAYER_WIDTH, PLAYER_HEIGHT = 8, 24
ENEMY_WIDTH, ENEMY_HEIGHT = 8, 24
FRUIT_WIDTH = 8
FRUIT_HEIGHT = 12

BELL_WIDTH = 6
BELL_HEIGHT = 11

CHILD_WIDTH = 8
CHILD_HEIGHT = 15

BACKGROUND_COLOR = (80, 0, 132)
PLAYER_COLOR = (223, 183, 85)
ENEMY_COLOR = (227, 151, 89)
FRUIT_COLOR_STATE_1 = (214, 92, 92)
FRUIT_COLOR_STATE_2 = (230, 250, 92)
FRUIT_COLOR_STATE_3 = (255, 92, 250)
FRUIT_COLOR_STATE_4 = (0, 92, 250)
FRUIT_COLOR = [
    FRUIT_COLOR_STATE_1,
    FRUIT_COLOR_STATE_2,
    FRUIT_COLOR_STATE_3,
    FRUIT_COLOR_STATE_4,
]
COCONUT_COLOR = PLATFORM_COLOR = (162, 98, 33)
LADDER_COLOR = (129, 78, 26)
BELL_COLOR = (210, 164, 74)

PLAYER_START_X, PLAYER_START_Y = 23, 148
MOVEMENT_SPEED = 1

LEFT_CLIP = 16
RIGHT_CLIP = 144

COCONUT_WIDTH = 3
COCONUT_HEIGHT = 4


# -------- Entity Classes --------
class Entity(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array


class PlayerState(NamedTuple):
    # Player position
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    orientation: chex.Array
    height: chex.Array
    # crouching
    is_crouching: chex.Array
    # jumping
    is_jumping: chex.Array
    jump_base_y: chex.Array
    jump_counter: chex.Array
    jump_orientation: chex.Array
    landing_base_y: chex.Array
    # climbing
    is_climbing: chex.Array
    climb_base_y: chex.Array
    climb_counter: chex.Array
    cooldown_counter: chex.Array
    # other
    is_crashing: chex.Array
    chrash_timer: chex.Array
    punch_left: chex.Array
    punch_right: chex.Array
    last_stood_on_platform_y: chex.Array


class LevelState(NamedTuple):
    timer: chex.Array
    platform_positions: chex.Array
    platform_sizes: chex.Array
    ladder_positions: chex.Array
    ladder_sizes: chex.Array
    fruit_positions: chex.Array
    fruit_actives: chex.Array
    fruit_stages: chex.Array
    bell_position: chex.Array
    bell_timer: chex.Array
    child_position: chex.Array
    child_velocity: chex.Array
    child_timer: chex.Array
    falling_coco_position: chex.Array
    falling_coco_dropping: chex.Array
    falling_coco_counter: chex.Array
    falling_coco_skip_update: chex.Array
    step_counter: chex.Array


class GameState(NamedTuple):
    player: PlayerState
    level: LevelState
    score: chex.Array
    current_level: chex.Array
    level_finished: chex.Array
    levelup_timer: chex.Array
    reset_coords: chex.Array
    levelup: chex.Array
    lives: chex.Array


# Level Constants
LADDER_HEIGHT = jnp.array(35)
LADDER_WIDTH = jnp.array(8)
P_HEIGHT = jnp.array(4)

LEVEL_1 = Kangaroo_Level_1
LEVEL_2 = Kangaroo_Level_2
LEVEL_3 = Kangaroo_Level_3


# -------- Keyboard Inputs --------
def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]
    reset = keys[pygame.K_r]

    if reset:
        return jnp.array(RESET)

    x, y = 0, 0
    if up and not down:
        y = 1
    elif not up and down:
        y = -1

    if left and not right:
        x = -1
    elif not left and right:
        x = 1

    if fire:
        if x == -1 and y == -1:
            return jnp.array(DOWNLEFTFIRE)
        elif x == -1 and y == 1:
            return jnp.array(UPLEFTFIRE)
        elif x == 1 and y == -1:
            return jnp.array(DOWNRIGHTFIRE)
        elif x == 1 and y == 1:
            return jnp.array(UPRIGHTFIRE)
        elif x == 0 and y == -1:
            return jnp.array(DOWNFIRE)
        else:
            return jnp.array(FIRE)
    else:
        if x == -1 and y == -1:
            return jnp.array(DOWNLEFT)
        elif x == -1 and y == 1:
            return jnp.array(UPLEFT)
        elif x == 1 and y == -1:
            return jnp.array(DOWNRIGHT)
        elif x == 1 and y == 1:
            return jnp.array(UPRIGHT)
        elif x == -1:
            return jnp.array(LEFT)
        elif x == 1:
            return jnp.array(RIGHT)
        elif y == -1:
            return jnp.array(DOWN)
        elif y == 1:
            return jnp.array(UP)

    return jnp.array(NOOP)


# TODO: used due to the JAX JIT requirement for same size arrays in conditional branches (see padding functions)
def is_valid_platform(platform_position: chex.Array) -> chex.Array:
    """Check if a platform position is valid (not padding)."""
    return platform_position[0] != -1


@partial(jax.jit, static_argnums=())
def get_platforms_below_player(state: GameState, y_offset=0) -> chex.Array:
    """Returns array of booleans indicating if player is on a platform."""
    player_x = state.player.x
    player_y = state.player.y + y_offset
    ph = state.player.height
    pw = PLAYER_WIDTH
    player_bottom_y = player_y + ph

    level_constants = get_level_constants(state.current_level)

    platform_positions = level_constants.platform_positions  # [N, 2] array of (x, y)
    platform_sizes = level_constants.platform_sizes  # [N, 2] array of (width, height)

    def calculate_lower_platform_index(i, carry):
        # Unpack the carry tuple
        prev_lowest_diff = carry[0]
        best_index = carry[1]

        platform_x = platform_positions[i, 0]
        platform_width = platform_sizes[i, 0]
        platform_y = platform_positions[i, 1]

        player_is_within_platform_x = jnp.logical_and(
            (player_x + pw) >= platform_x, player_x <= (platform_x + platform_width)
        )

        platform_is_below_player = player_bottom_y <= platform_y
        diff_to_platform_i = jnp.where(
            platform_is_below_player, platform_y - player_bottom_y, 1000
        )

        # Update both the lowest difference and the corresponding index
        new_best_diff = jnp.where(
            player_is_within_platform_x
            & platform_is_below_player
            & (diff_to_platform_i < prev_lowest_diff),
            diff_to_platform_i,
            prev_lowest_diff,
        )

        new_index = jnp.where(
            player_is_within_platform_x
            & platform_is_below_player
            & (diff_to_platform_i < prev_lowest_diff),
            i,
            best_index,
        )

        is_valid = is_valid_platform(platform_positions[i])
        return jnp.where(
            is_valid,
            jnp.array([new_best_diff.astype(int), new_index.astype(int)]),
            carry,
        )

    # Initialize with (difference, index)
    y_diffs_and_indices_array = jax.lax.fori_loop(
        0,
        platform_positions.shape[0],
        calculate_lower_platform_index,
        jnp.array([int(1000), int(-1)]),
    )

    platform_under_player_index = y_diffs_and_indices_array[1].astype(int)
    return_value = jnp.zeros(platform_positions.shape[0], dtype=bool)

    return return_value.at[platform_under_player_index].set(True)


@partial(jax.jit, static_argnums=())
def entities_collide_with_threshold(
    e1_x: chex.Array,
    e1_y: chex.Array,
    e1_w: chex.Array,
    e1_h: chex.Array,
    e2_x: chex.Array,
    e2_y: chex.Array,
    e2_w: chex.Array,
    e2_h: chex.Array,
    threshold: chex.Array,
) -> chex.Array:
    """Returns True if rectangles overlap by at least threshold fraction. This only Checks for overlap in the x dimension."""
    overlap_start_x = jnp.maximum(e1_x, e2_x)
    overlap_end_x = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
    overlap_start_y = jnp.maximum(e1_y, e2_y)
    overlap_end_y = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

    # Calculate dimensions of overlap region
    overlap_width = overlap_end_x - overlap_start_x
    overlap_height = overlap_end_y - overlap_start_y

    # Calculate minimum required overlap area based on threshold
    min_required_overlap = e1_w * threshold

    # Check if overlap exceeds required threshold
    meets_threshold = overlap_width >= min_required_overlap

    return jax.lax.cond(
        jnp.any(jnp.array([overlap_width, overlap_height]) < 0),
        lambda _: jnp.array(False),
        lambda y: y,
        meets_threshold,
    )


@partial(jax.jit, static_argnums=())
def entities_collide(
    e1_x: chex.Array,
    e1_y: chex.Array,
    e1_w: chex.Array,
    e1_h: chex.Array,
    e2_x: chex.Array,
    e2_y: chex.Array,
    e2_w: chex.Array,
    e2_h: chex.Array,
) -> chex.Array:
    """
    Calls do_collide_with_threshold with a threshold of 0 and checks if two rectangles overlap.
    """
    return entities_collide_with_threshold(
        e1_x, e1_y, e1_w, e1_h, e2_x, e2_y, e2_w, e2_h, 0
    )


@partial(jax.jit, static_argnums=())
def player_is_above_ladder(
    state: GameState,
    threshold: float = 0.3,
    virtual_hitbox_height: float = 12.0,
) -> chex.Array:
    """Checks collision between a virtual hitbox below player and ladders."""

    level_constants = get_level_constants(state.current_level)

    def check_single_collision(i, collisions):
        ladder_pos = level_constants.ladder_positions[i]
        ladder_size = level_constants.ladder_sizes[i]

        collision = entities_collide_with_threshold(
            state.player.x,
            state.player.y + state.player.height,
            PLAYER_WIDTH,
            virtual_hitbox_height,
            ladder_pos[0],
            ladder_pos[1],
            ladder_size[0],
            ladder_size[1],
            threshold,
        )

        return collisions.at[i].set(collision)

    num_ladders = level_constants.ladder_positions.shape[0]
    initial_collisions = jnp.zeros(num_ladders, dtype=bool)
    return jax.lax.fori_loop(0, num_ladders, check_single_collision, initial_collisions)


@partial(jax.jit, static_argnums=())
def check_ladder_collisions(
    state: GameState, level_constants: LevelConstants, threshold: float = 0.3
) -> chex.Array:
    """Vectorized ladder collision checking."""

    def check_single_ladder(i, collisions):
        ladder_pos = level_constants.ladder_positions[i]
        ladder_size = level_constants.ladder_sizes[i]

        collision = entities_collide_with_threshold(
            state.player.x,
            state.player.y,
            PLAYER_WIDTH,
            state.player.height,
            ladder_pos[0],
            ladder_pos[1],
            ladder_size[0],
            ladder_size[1],
            threshold,
        )

        return collisions.at[i].set(collision)

    num_ladders = level_constants.ladder_positions.shape[0]
    collisions = jnp.zeros(num_ladders, dtype=bool)
    return jax.lax.fori_loop(0, num_ladders, check_single_ladder, collisions)


def player_is_on_ladder(
    state: GameState,
    ladder_pos: chex.Array,
    ladder_size: chex.Array,
    threshold: float = 0.3,
) -> chex.Array:
    """
    Checks the collision of the player with a ladder. <threshold>% of the players surface area have to overlap with the ladder.
    """
    return entities_collide_with_threshold(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        ladder_pos[0],
        ladder_pos[1],
        ladder_size[0],
        ladder_size[1],
        threshold,
    )


@partial(jax.jit, static_argnums=())
# -------- Jumping and Climbing --------
def player_jump_controller(
    state: GameState, jump_pressed: chex.Array, ladder_intersect: chex.Array
):
    """
    Schedule:
      tick  8: total offset = -12
      tick 16: still -12 (just sprite/hitbox change)
      tick 24: total offset = -24
      tick 32: total offset = -12  (moved up +12 from -24)
      tick 40: total offset =  0   (moved up +12 from -12) -> jump ends
    """
    player_y = state.player.y
    jump_counter = state.player.jump_counter
    is_jumping = state.player.is_jumping

    cooldown_condition = state.player.cooldown_counter > 0
    jump_start = jump_pressed & ~is_jumping & ~ladder_intersect & ~cooldown_condition

    # Update jump state on start
    jump_counter = jnp.where(jump_start, 0, jump_counter)
    jump_orientation = jnp.where(
        jump_start, state.player.orientation, state.player.jump_orientation
    )
    jump_base_y = jnp.where(jump_start, player_y, state.player.jump_base_y)
    new_landing_base_y = jump_base_y
    # check if player is on/above a new platform and change jump_base_y accordingly

    platform_y_below_player = get_y_of_platform_below_player(state)

    # find a new potential landing_base if player is above a higher platform
    new_landing_base_y = jnp.where(
        is_jumping
        & ((platform_y_below_player - PLAYER_HEIGHT) < jump_base_y)
        & ~jump_start,
        platform_y_below_player - PLAYER_HEIGHT,
        new_landing_base_y,
    )

    is_jumping = is_jumping | jump_start

    # Update counter if jumping
    jump_counter = jnp.where(is_jumping, jump_counter + 1, jump_counter)

    # Calculate vertical offset based on jump phase
    def offset_for(count):
        conditions = [
            (count <= 8),
            (count < 16),
            (count <= 24),
            (count <= 32),
            (count < 41),
        ]
        values = [
            0,
            -8,
            -8,
            -16,
            -8,
        ]
        return jnp.select(conditions, values, default=0)

    # check if player is on a new platform and cancel jump if so
    jump_cancel = (
        is_jumping
        & (player_y >= new_landing_base_y)
        & (new_landing_base_y < jump_base_y)
        & (jump_counter > 32)
    )
    jump_counter = jnp.where(jump_cancel, 40, jump_counter)
    jump_base_y = jnp.where(jump_cancel, new_landing_base_y, jump_base_y)
    new_y = jnp.where(jump_cancel, new_landing_base_y, player_y)
    new_cooldown_counter = jnp.where(jump_cancel, 8, state.player.cooldown_counter)

    total_offset = offset_for(jump_counter)
    new_y = jnp.where(is_jumping & ~jump_cancel, jump_base_y + total_offset, new_y)

    # Check for jump completion
    jump_complete = jump_counter >= 41
    is_jumping = jnp.where(jump_complete, False, is_jumping)
    jump_counter = jnp.where(jump_complete, 0, jump_counter)

    return_value = (
        new_y,
        jump_counter,
        is_jumping,
        jump_base_y,
        new_landing_base_y,
        jump_orientation,
        new_cooldown_counter,
    )

    return jax.lax.cond(
        state.levelup_timer == 0,
        lambda: return_value,
        lambda: (
            state.player.y,
            state.player.jump_counter,
            state.player.is_jumping,
            state.player.jump_base_y,
            state.player.landing_base_y,
            state.player.jump_orientation,
            state.player.cooldown_counter,
        ),
    )


@partial(jax.jit, static_argnums=())
def player_climb_controller(
    state: GameState,
    y: chex.Array,
    press_up: chex.Array,
    press_down: chex.Array,
    ladder_intersect: chex.Array,
) -> tuple[Array, Array, Array, Array, Array]:

    # Ladder Below Collision
    ladder_intersect_below = jnp.any(player_is_above_ladder(state))

    new_y = y
    is_climbing = state.player.is_climbing
    is_climbing = jnp.where(state.player.is_jumping, False, is_climbing)

    climb_counter = state.player.climb_counter

    cooldown_over = state.player.cooldown_counter <= 0

    climb_start = (
        press_up
        & ~is_climbing
        & ladder_intersect
        & ~state.player.is_jumping
        & cooldown_over
    )
    climb_start_downward = (
        press_down
        & ~is_climbing
        & ladder_intersect_below
        & ~state.player.is_jumping
        & cooldown_over
    )

    is_climbing = is_climbing | climb_start | climb_start_downward

    climb_counter = jnp.where(climb_start | climb_start_downward, 0, climb_counter)

    climb_base_y = jnp.where(climb_start, new_y, state.player.climb_base_y)

    climb_base_y = jnp.where(
        climb_start_downward,
        get_y_of_platform_below_player(state, 1) - PLAYER_HEIGHT,
        climb_base_y,
    )

    new_y = jnp.where(climb_start, new_y - 8, new_y)
    new_y = jnp.where(climb_start_downward, new_y + 8, new_y)

    climb_counter = jnp.where(is_climbing, climb_counter + 1, climb_counter)

    climb_up = jnp.logical_and(press_up, is_climbing)
    climb_down = jnp.logical_and(press_down, is_climbing)

    new_y = jnp.where(
        jnp.logical_and(climb_up, jnp.equal(climb_counter, 19)), new_y - 8, new_y
    )
    new_y = jnp.where(
        jnp.logical_and(climb_down, jnp.equal(climb_counter, 19)), new_y + 8, new_y
    )

    set_new_climb_base = (
        climb_up
        & ((get_y_of_platform_below_player(state) - state.player.height) >= new_y)
        & ladder_intersect
    )
    climb_base_y = jnp.where(
        set_new_climb_base,  # when player is on a new platform but still climbing up
        get_y_of_platform_below_player(state) - PLAYER_HEIGHT,
        climb_base_y,
    )
    # Check if not climbing anymore -> bottom of ladder
    climb_stop = is_climbing & (new_y >= climb_base_y) & ~climb_start_downward

    is_climbing = jnp.where(climb_stop, False, is_climbing)

    # Check if not climbing anymore -> top of ladder
    is_climbing = jnp.where(ladder_intersect | climb_start_downward, is_climbing, False)

    climb_counter = jnp.where(climb_counter >= 19, 0, climb_counter)
    cooldown_counter = jnp.where(
        climb_stop | set_new_climb_base,
        15,
        jnp.where(
            state.player.cooldown_counter > 0, state.player.cooldown_counter - 1, 0
        ),
    )

    return_value = (new_y, is_climbing, climb_base_y, climb_counter, cooldown_counter)

    return jax.lax.cond(
        state.levelup_timer == 0,
        lambda: return_value,
        lambda: (
            state.player.y,
            state.player.is_climbing,
            state.player.climb_base_y,
            state.player.climb_counter,
            state.player.cooldown_counter,
        ),
    )


# -------- Player Height --------
@partial(jax.jit, static_argnums=())
def player_height_controller(
    is_jumping: chex.Array,
    jump_counter: chex.Array,
    is_crouching: chex.Array,
) -> chex.Array:
    """
    Jump-based height changes:
      -  0..15 ticks => height=24 (normal)
      - 16..23 ticks => height=15 (small sprite)
      - 24..39 ticks => height=23 (stretched jump sprite)
      - >= 40        => height=24 (back on ground)

    If pressing DOWN *and not jumping*, override => 16 for ducking.
    """

    def jump_based_height(count):
        conditions = [
            (count < 16),
            (count < 24),
            (count < 40),
        ]
        values = [
            24,
            15,
            23,
        ]
        return jnp.select(conditions, values, default=24)

    candidate_height = jump_based_height(jump_counter)
    height_if_jumping = jnp.where(is_jumping, candidate_height, 24)

    is_crouching = jnp.logical_and(is_crouching, jnp.logical_not(is_jumping))

    new_height = jnp.where(is_crouching, 16, height_if_jumping)
    return new_height


@partial(jax.jit, static_argnums=())
def get_y_of_platform_below_player(state: GameState, y_offset=0) -> chex.Array:
    """Gets the y-position of the next platform below the player."""

    level_constants = get_level_constants(state.current_level)

    platform_bands: jax.Array = get_platforms_below_player(state, y_offset)
    platform_ys = level_constants.platform_positions[:, 1]

    def find_next_platform(i, current_platform_y):
        is_valid = is_valid_platform(level_constants.platform_positions[i])
        is_in_band = jnp.logical_and(platform_bands[i], is_valid)
        return jnp.where(
            is_in_band & (platform_ys[i] < current_platform_y),
            platform_ys[i],
            current_platform_y,
        )

    initial_y = platform_ys[0]
    return jax.lax.fori_loop(0, platform_ys.shape[0], find_next_platform, initial_y)


@partial(jax.jit, static_argnums=())
def fruits_step(state: GameState) -> Tuple[chex.Array, chex.Array]:
    """Handles fruit collection and scoring."""

    def check_fruit(i, carry):
        score, actives = carry
        fruit_position = state.level.fruit_positions[i]

        fruit_collision = entities_collide(
            state.player.x,
            state.player.y,
            PLAYER_WIDTH,
            state.player.height,
            fruit_position[0],
            fruit_position[1],
            FRUIT_WIDTH,
            FRUIT_HEIGHT,
        )

        collision_condition = jnp.logical_and(fruit_collision, actives[i])

        new_score = jnp.where(
            collision_condition,
            score + (200 * state.level.fruit_stages[i] + 200),
            score,
        )
        new_actives = actives.at[i].set(
            jnp.where(collision_condition, False, actives[i])
        )

        return new_score, new_actives

    bell_collision = entities_collide(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        state.level.bell_position[0],
        state.level.bell_position[1],
        BELL_WIDTH,
        BELL_HEIGHT,
    )

    RESPAWN_AFTER_TICKS = 40

    counter = state.level.bell_timer
    counter_start = bell_collision & (counter == 0)
    counter = jnp.where(counter_start, 1, counter)
    counter = jnp.where(counter > 0, counter + 1, counter)
    counter = jnp.where(counter == RESPAWN_AFTER_TICKS + 1, 0, counter)
    respawn_timer_done = counter == RESPAWN_AFTER_TICKS

    initial_score = jnp.array(0)
    initial_actives = state.level.fruit_actives

    new_score, new_activations = jax.lax.fori_loop(
        0,
        state.level.fruit_actives.shape[0],
        check_fruit,
        (initial_score, initial_actives),
    )

    stage_fruit_1 = jnp.where(
        respawn_timer_done & (state.level.fruit_actives[0] == False),
        jnp.clip(state.level.fruit_stages[0] + 1, 0, 4),
        state.level.fruit_stages[0],
    )
    stage_fruit_2 = jnp.where(
        respawn_timer_done & (state.level.fruit_actives[1] == False),
        jnp.clip(state.level.fruit_stages[1] + 1, 0, 4),
        state.level.fruit_stages[1],
    )
    stage_fruit_3 = jnp.where(
        respawn_timer_done & (state.level.fruit_actives[2] == False),
        jnp.clip(state.level.fruit_stages[2] + 1, 0, 4),
        state.level.fruit_stages[2],
    )

    new_stages = jnp.array([stage_fruit_1, stage_fruit_2, stage_fruit_3])

    activations = jax.lax.cond(
        respawn_timer_done,
        lambda: jnp.less_equal(new_stages, jnp.array([3, 3, 3])),
        lambda: jnp.array([new_activations[0], new_activations[1], new_activations[2]]),
    )

    return new_score, activations, new_stages, counter


@partial(jax.jit, static_argnums=())
def child_step(state: GameState) -> Tuple[chex.Array]:

    RESET_TIMER_AFTER = 50

    counter = state.level.child_timer
    counter = counter + 1
    counter = jnp.where(counter > RESET_TIMER_AFTER, 0, counter)
    reset = counter == RESET_TIMER_AFTER

    child_velocity = state.level.child_velocity
    new_child_velocity = jnp.where(reset, child_velocity * -1, child_velocity)

    new_child_x = jnp.where(
        state.levelup_timer == 0,
        jnp.where(
            (counter % 5) == 0,
            state.level.child_position[0] + new_child_velocity,
            state.level.child_position[0],
        ),
        state.level.child_position[0],
    )
    new_child_y = state.level.child_position[1]
    new_child_timer = counter

    return new_child_timer, new_child_x, new_child_y, new_child_velocity


def pad_array(arr: jax.Array, target_size: int):
    """Pads a 2D array with -1s to reach target size in first dimension."""
    current_size = arr.shape[0]

    return jnp.pad(
        arr,
        ((0, target_size - current_size), (0, 0)),
        mode="constant",
        constant_values=-1,
    )


def pad_to_size(level_constants: LevelConstants, max_platforms: int):
    """Pads all arrays in level constants to specified size."""
    return LevelConstants(
        ladder_positions=pad_array(level_constants.ladder_positions, max_platforms),
        ladder_sizes=pad_array(level_constants.ladder_sizes, max_platforms),
        platform_positions=pad_array(level_constants.platform_positions, max_platforms),
        platform_sizes=pad_array(level_constants.platform_sizes, max_platforms),
        fruit_positions=level_constants.fruit_positions,
        bell_position=level_constants.bell_position,
        child_position=level_constants.child_position,
    )


@partial(jax.jit, static_argnums=())
def get_level_constants(current_level):
    """Returns constants for the current level."""
    max_platforms = 20

    # Pad each level's arrays to max size
    level1_padded = pad_to_size(LEVEL_1, max_platforms)
    level2_padded = pad_to_size(LEVEL_2, max_platforms)
    level3_padded = pad_to_size(LEVEL_3, max_platforms)

    return jax.lax.cond(
        current_level == 1,
        lambda _: level1_padded,
        lambda _: jax.lax.cond(
            current_level == 2,
            lambda _: level2_padded,
            lambda _: level3_padded,
            operand=None,
        ),
        operand=None,
    )


@partial(jax.jit, static_argnums=())
def player_step(state: GameState, action: chex.Array):
    """Main player movement and state update function."""
    level_constants = get_level_constants(state.current_level)
    x, y = state.player.x, state.player.y
    old_height = state.player.height
    old_orientation = state.player.orientation

    # Get inputs
    press_right = jnp.any(
        jnp.array([action == RIGHT, action == UPRIGHT, action == DOWNRIGHT])
    )

    press_left = jnp.any(
        jnp.array([action == LEFT, action == UPLEFT, action == DOWNLEFT])
    )

    press_up = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT]))

    press_fire = jnp.any(
        jnp.array(
            [
                action == FIRE,
                action == DOWNFIRE,
                action == UPLEFTFIRE,
                action == UPRIGHTFIRE,
                action == DOWNLEFTFIRE,
                action == DOWNRIGHTFIRE,
            ]
        )
    )

    press_down_fire = jnp.any(jnp.array(action == DOWNFIRE))

    press_down = jnp.any(
        jnp.array([action == DOWN, action == DOWNLEFT, action == DOWNRIGHT])
    )

    press_down = jnp.where(state.player.is_jumping, False, press_down)
    press_fire = jnp.where(state.player.is_jumping, False, press_fire)
    press_fire = jnp.where(state.player.is_climbing, False, press_fire)
    press_fire = jnp.where(press_down_fire, False, press_fire)

    press_up = jnp.where(press_down_fire, False, press_up)

    # Forbid left/right movement while climbing
    press_right = jnp.where(state.player.is_climbing, False, press_right)
    press_left = jnp.where(state.player.is_climbing, False, press_left)

    is_looking_left = state.player.orientation == -1
    is_looking_right = state.player.orientation == 1
    is_punching_left = (
        jnp.logical_and(press_fire, is_looking_left) & ~state.player.is_crashing
    )
    is_punching_right = (
        jnp.logical_and(press_fire, is_looking_right) & ~state.player.is_crashing
    )

    # check for any collision with standard threshold
    ladder_intersect_thresh = jnp.any(check_ladder_collisions(state, level_constants))

    ladder_intersect_no_thresh = jnp.any(
        check_ladder_collisions(state, level_constants, 0)
    )

    ladder_intersect = jnp.where(
        state.player.is_climbing, ladder_intersect_no_thresh, ladder_intersect_thresh
    )

    # Jump controller
    (
        new_y,
        new_jump_counter,
        new_is_jumping,
        new_jump_base_y,
        new_landing_base_y,
        new_jump_orientation,
        new_cooldown_counter,
    ) = player_jump_controller(state, press_up, ladder_intersect)

    # Climb controller
    (
        new_y,
        new_is_climbing,
        new_climb_base_y,
        new_climb_counter,
        new_cooldown_counter,
    ) = player_climb_controller(state, new_y, press_up, press_down, ladder_intersect)

    new_is_crouching = press_down & ~new_is_climbing & ~new_is_jumping

    # Calculate horizontal velocity
    candidate_vel_x = jnp.where(
        new_is_crouching,
        0,
        jnp.where(
            press_left, -MOVEMENT_SPEED, jnp.where(press_right, MOVEMENT_SPEED, 0)
        ),
    )

    # Check Orientation (Left/Right)
    # if standing still, keep the old orientation
    standing_still = jnp.equal(candidate_vel_x, 0)
    new_orientation = jnp.sign(candidate_vel_x)
    new_orientation = jnp.where(standing_still, old_orientation, new_orientation)

    stop_in_air = jnp.logical_and(
        new_is_jumping, state.player.jump_orientation != new_orientation
    )

    # Stop Jump when orientation changes mid air
    vel_x = jnp.where(stop_in_air, 0, candidate_vel_x)

    # Height controller
    new_player_height = player_height_controller(
        is_jumping=new_is_jumping,
        jump_counter=new_jump_counter,
        is_crouching=new_is_crouching,
    )
    new_player_height = jnp.where(
        (state.levelup_timer > 0) | state.player.is_crashing,
        PLAYER_HEIGHT,
        new_player_height,
    )
    # If height changes, shift the player's top so the bottom remains consistent
    dy = old_height - new_player_height
    new_y = new_y + dy

    # x-axis movement
    x = jnp.where(
        state.player.is_crashing | state.levelup_timer != 0,
        x,
        jnp.clip(x + vel_x, LEFT_CLIP, RIGHT_CLIP - PLAYER_WIDTH),
    )

    # y-axis movement
    platform_bools: jax.Array = get_platforms_below_player(state)
    platform_ys: jax.Array = level_constants.platform_positions[:, 1]

    def get_platform_dependent_y(i, curr_y):
        is_valid = is_valid_platform(level_constants.platform_positions[i])
        platform_y = jnp.where(
            jnp.logical_and(platform_bools[i], is_valid),
            jnp.where(
                ~state.player.is_climbing & new_is_climbing & press_down,
                new_y,
                jnp.clip(new_y, 0, platform_ys[i] - new_player_height),
            ),
            curr_y,
        )
        return platform_y

    # iterate the platform bools and check which is true, then return the corresponding y value
    platform_dependent_y = jax.lax.fori_loop(
        0, len(platform_bools), get_platform_dependent_y, new_y
    )

    y = jnp.where(
        state.player.is_crashing,
        jnp.where((y + new_player_height) > SCREEN_HEIGHT, y, y + 2),
        platform_dependent_y,
    )

    # check if player reached the final platform
    final_platform_y = 28
    player_on_last_platform = (new_y + new_player_height) == final_platform_y
    level_finished = (
        player_on_last_platform & ~state.level_finished & (state.levelup_timer == 0)
    )

    y = jnp.where(state.levelup_timer == 0, y, state.player.y)

    # Reset X and Y when going to next level
    x = jnp.where(state.reset_coords, PLAYER_START_X, x)
    y = jnp.where(state.reset_coords, PLAYER_START_Y, y)

    return (
        x,
        y,
        vel_x,
        new_is_crouching,
        new_is_jumping,
        new_is_climbing,
        new_jump_counter,
        new_orientation,
        new_jump_base_y,
        new_landing_base_y,
        new_player_height,
        new_jump_orientation,
        new_climb_base_y,
        new_climb_counter,
        is_punching_left,
        is_punching_right,
        new_cooldown_counter,
        level_finished,
    )


@partial(jax.jit, static_argnums=())
def timer_controller(state: GameState):
    return jnp.where(
        state.level.step_counter == 255, state.level.timer - 100, state.level.timer
    )


@partial(jax.jit, static_argnums=())
def next_level(state: GameState):

    RESET_AFTER_TICKS = 256

    counter = state.levelup_timer
    counter_start = state.level_finished & (counter == 0)
    counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
    reset_timer_done = counter == RESET_AFTER_TICKS
    counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

    reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
    levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

    current_level = jnp.where(levelup, state.current_level + 1, state.current_level)

    return current_level, counter, reset_coords, levelup


@partial(jax.jit, static_argnums=())
def lives_controller(state: GameState):
    # timer check
    is_time_over = state.level.timer <= 0

    new_last_stood_on_platform_y = jnp.where(
        get_y_of_platform_below_player(state) == (state.player.y + state.player.height),
        get_y_of_platform_below_player(state),
        state.player.last_stood_on_platform_y,
    )

    # platform_drop_check()

    y_of_platform_below_player = get_y_of_platform_below_player(state)
    player_is_falling = (
        (state.player.y + state.player.height) == state.player.last_stood_on_platform_y
    ) & (y_of_platform_below_player > state.player.last_stood_on_platform_y)

    # monkey touch check

    # coconut touch check

    crashed_falling_coco = entities_collide(
        state.player.x,
        state.player.y,
        PLAYER_WIDTH,
        state.player.height,
        state.level.falling_coco_position[0],
        state.level.falling_coco_position[1],
        COCONUT_WIDTH,
        COCONUT_HEIGHT,
    )

    remove_live = (
        is_time_over | player_is_falling | crashed_falling_coco
    ) & ~state.player.is_crashing
    new_is_crashing = jnp.where(remove_live, True, state.player.is_crashing)

    start_timer = (
        state.player.is_crashing
        & (state.player.chrash_timer == 0)
        & ((state.player.y + state.player.height) > SCREEN_HEIGHT)
    )

    # start counter
    RESPAWN_AFTER_TICKS = 40

    counter = state.player.chrash_timer
    counter_start = start_timer
    counter = jnp.where(counter_start, 1, counter)
    counter = jnp.where(counter > 0, counter + 1, counter)
    counter = jnp.where(counter == RESPAWN_AFTER_TICKS + 1, 0, counter)
    crash_timer_done = counter == RESPAWN_AFTER_TICKS

    new_is_crashing = jnp.where(crash_timer_done, False, new_is_crashing)

    return (
        jnp.where(remove_live, state.lives - 1, state.lives),
        new_is_crashing,
        counter,
        crash_timer_done,
        new_last_stood_on_platform_y,
    )


@partial(jax.jit, static_argnums=(0))
def falling_coconut_controller(state: GameState):
    falling_coco_exists = (state.level.falling_coco_position[0] != 13) | (
        state.level.falling_coco_position[1] != -1
    )

    # update coco position
    spawn_new_coco = ~falling_coco_exists & (state.level.step_counter == 255)

    update_positions = ~state.level.falling_coco_skip_update & (
        ((state.level.step_counter % 8) == 0) | spawn_new_coco
    )

    # detect if coco is above player and switch from x-following state to dropping state
    coco_first_time_above_player = (
        ~state.level.falling_coco_dropping
        & falling_coco_exists
        & (
            ((state.level.falling_coco_position[0] + COCONUT_WIDTH) > state.player.x)
            & (state.level.falling_coco_position[0] < (state.player.x + PLAYER_WIDTH))
        )
        & update_positions
    )

    new_falling_coco_dropping = jnp.where(
        coco_first_time_above_player, True, state.level.falling_coco_dropping
    )

    new_falling_coco_skip_update = coco_first_time_above_player

    # reset skip update
    new_falling_coco_skip_update = jnp.where(
        state.level.falling_coco_skip_update
        & (((state.level.step_counter % 8) == 0) | spawn_new_coco),
        False,
        new_falling_coco_skip_update | state.level.falling_coco_skip_update,
    )

    new_falling_coco_counter = jnp.where(
        update_positions,
        jnp.where(
            spawn_new_coco,
            0,
            jnp.where(
                state.level.falling_coco_dropping & update_positions,
                state.level.falling_coco_counter + 1,
                state.level.falling_coco_counter,
            ),
        ),
        state.level.falling_coco_counter,
    )

    reset_coco = new_falling_coco_counter > 20
    new_falling_coco_counter = jnp.where(reset_coco, 0, new_falling_coco_counter)

    new_falling_coco_dropping = jnp.where(reset_coco, False, new_falling_coco_dropping)

    new_falling_coco_position_x = jnp.where(
        update_positions
        & ~state.level.falling_coco_dropping
        & (falling_coco_exists | spawn_new_coco),
        state.level.falling_coco_position[0] + 2,
        state.level.falling_coco_position[0],
    )
    new_falling_coco_position_y = jnp.where(
        update_positions & (falling_coco_exists | spawn_new_coco),
        8 * new_falling_coco_counter + 8,
        state.level.falling_coco_position[1],
    )

    new_falling_coco_position = jnp.where(
        reset_coco,
        jnp.array([13, -1]),
        jnp.array([new_falling_coco_position_x, new_falling_coco_position_y]),
    )

    return (
        new_falling_coco_position,
        new_falling_coco_dropping,
        new_falling_coco_counter,
        new_falling_coco_skip_update,
    )


# -------- Game Interface for Reset and Step --------
class Game:
    def __init__(self, frameskip: int = 1):
        self.frameskip = frameskip

    def reset(self, next_level=1) -> GameState:

        next_level = jnp.clip(next_level, 1, 3)
        level_constants: LevelConstants = get_level_constants(next_level)

        return GameState(
            player=PlayerState(
                x=jnp.array(PLAYER_START_X),
                y=jnp.array(PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(PLAYER_START_Y),
                landing_base_y=jnp.array(PLAYER_START_Y),
                height=jnp.array(PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
                chrash_timer=jnp.array(0),
                is_crashing=jnp.array(False),
                last_stood_on_platform_y=jnp.array(1000),
            ),
            level=LevelState(
                bell_position=level_constants.bell_position,
                bell_timer=jnp.array(0),
                fruit_positions=level_constants.fruit_positions,
                fruit_actives=jnp.ones(3, dtype=jnp.bool_),
                fruit_stages=jnp.zeros(3, dtype=jnp.int32),
                ladder_positions=level_constants.ladder_positions,
                ladder_sizes=level_constants.ladder_sizes,
                platform_positions=level_constants.platform_positions,
                platform_sizes=level_constants.platform_sizes,
                child_position=level_constants.child_position,
                child_timer=jnp.array(0),
                child_velocity=jnp.array(1),
                timer=jnp.array(2000),
                falling_coco_position=jnp.array([13, -1]),
                falling_coco_dropping=jnp.array(False),
                falling_coco_counter=jnp.array(0),
                falling_coco_skip_update=jnp.array(False),
                step_counter=jnp.array(0),
            ),
            score=jnp.array(0),
            current_level=jnp.array(next_level),
            level_finished=jnp.array(False),
            levelup_timer=jnp.array(0),
            reset_coords=jnp.array(False),
            levelup=jnp.array(False),
            lives=3,
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: chex.Array) -> GameState:
        reset_cond = jnp.any(jnp.array([action == RESET]))

        # Update player state
        (
            player_x,
            player_y,
            vel_x,
            is_crouching,
            is_jumping,
            is_climbing,
            jump_counter,
            orientation,
            jump_base_y,
            landing_base_y,
            new_player_height,
            new_jump_orientation,
            climb_base_y,
            climb_counter,
            punch_left,
            punch_right,
            cooldown_counter,
            level_finished,
        ) = player_step(state, action)

        new_current_level, new_levelup_timer, new_reset_coords, new_levelup = (
            next_level(state)
        )

        # Handle fruit collection
        score_addition, new_actives, new_fruit_stages, bell_timer = fruits_step(state)
        child_timer, new_child_x, new_child_y, new_child_velocity = child_step(state)

        # Handle Main Timer
        new_main_timer = timer_controller(state)

        (
            new_falling_coco_position,
            new_falling_coco_dropping,
            new_falling_coco_counter,
            new_falling_coco_skip_update,
        ) = falling_coconut_controller(state)

        (
            new_lives,
            new_is_crashing,
            crash_timer,
            crash_timer_done,
            new_last_stood_on_platform_y,
        ) = lives_controller(state)

        # reset_current_level_progress()

        # add score if levelup from lvl3 to lvl1
        score_addition = jax.lax.cond(
            new_current_level == 4,
            lambda: score_addition + 1400,
            lambda: score_addition,
        )
        new_current_level = jnp.where(new_current_level == 4, 1, new_current_level)

        new_level_state = jax.lax.cond(
            new_levelup,
            lambda: self.reset(new_current_level).level,
            lambda: jax.lax.cond(
                crash_timer_done,
                lambda: self.reset(state.current_level).level,
                lambda: LevelState(
                    bell_position=state.level.bell_position,
                    fruit_positions=state.level.fruit_positions,
                    ladder_positions=state.level.ladder_positions,
                    ladder_sizes=state.level.ladder_sizes,
                    platform_positions=state.level.platform_positions,
                    platform_sizes=state.level.platform_sizes,
                    child_position=jnp.array([new_child_x, new_child_y]),
                    timer=new_main_timer,
                    bell_timer=bell_timer,
                    child_timer=child_timer,
                    child_velocity=new_child_velocity,
                    fruit_actives=new_actives,
                    fruit_stages=new_fruit_stages,
                    falling_coco_position=new_falling_coco_position,
                    falling_coco_dropping=new_falling_coco_dropping,
                    falling_coco_counter=new_falling_coco_counter,
                    falling_coco_skip_update=new_falling_coco_skip_update,
                    step_counter=(state.level.step_counter + 1) % 256,
                ),
            ),
        )

        new_player_state = jax.lax.cond(
            crash_timer_done,
            lambda: self.reset(state.current_level).player,
            lambda: PlayerState(
                x=player_x,
                y=player_y,
                vel_x=vel_x,
                is_crouching=is_crouching,
                is_jumping=is_jumping,
                is_climbing=is_climbing,
                jump_counter=jump_counter,
                orientation=orientation,
                jump_base_y=jump_base_y,
                landing_base_y=landing_base_y,
                height=new_player_height,
                jump_orientation=new_jump_orientation,
                climb_base_y=climb_base_y,
                climb_counter=climb_counter,
                punch_left=punch_left,
                punch_right=punch_right,
                cooldown_counter=cooldown_counter,
                chrash_timer=crash_timer,
                is_crashing=new_is_crashing,
                last_stood_on_platform_y=new_last_stood_on_platform_y,
            ),
        )

        # jax.debug.print(
        #     "new_is_crashing={nic} | crash_timer={ct} | crash_timer_done={ctd} | new_last_stood_on_platform_y={ly}",
        #     nic=new_is_crashing,
        #     ct=crash_timer,
        #     ctd=crash_timer_done,
        #     ly=new_last_stood_on_platform_y,
        # )

        return jax.lax.cond(
            reset_cond,
            lambda: self.reset(state.current_level),
            lambda: GameState(
                player=new_player_state,
                level=new_level_state,
                score=state.score + score_addition,
                current_level=new_current_level,
                level_finished=level_finished,
                levelup_timer=new_levelup_timer,
                reset_coords=new_reset_coords,
                levelup=new_levelup,
                lives=new_lives,
            ),
        )


# ----------------------------------------------------------------
# -------- Ad-Hoc Rendering (to be redone by other group) --------
# ----------------------------------------------------------------


class Renderer:
    def __init__(self):
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH * RENDER_SCALE_FACTOR, SCREEN_HEIGHT * RENDER_SCALE_FACTOR)
        )
        pygame.display.set_caption("Kangaroo")

    def render(self, state: GameState):
        self.screen.fill(BACKGROUND_COLOR)

        # Walls
        pygame.draw.rect(
            self.screen,
            PLATFORM_COLOR,
            (
                0,
                0,
                LEFT_CLIP * RENDER_SCALE_FACTOR,
                SCREEN_HEIGHT * RENDER_SCALE_FACTOR,
            ),
        )
        pygame.draw.rect(
            self.screen,
            PLATFORM_COLOR,
            (
                RIGHT_CLIP * RENDER_SCALE_FACTOR,
                0,
                16 * RENDER_SCALE_FACTOR,
                SCREEN_HEIGHT * RENDER_SCALE_FACTOR,
            ),
        )

        # Get level constants for current level
        level_constants = get_level_constants(state.current_level)

        # Draw ladders
        for i in range(level_constants.ladder_positions.shape[0]):
            pos = level_constants.ladder_positions[i]
            size = level_constants.ladder_sizes[i]
            pygame.draw.rect(
                self.screen,
                LADDER_COLOR,
                (
                    int(pos[0]) * RENDER_SCALE_FACTOR,
                    int(pos[1]) * RENDER_SCALE_FACTOR,
                    int(size[0]) * RENDER_SCALE_FACTOR,
                    int(size[1]) * RENDER_SCALE_FACTOR,
                ),
            )
            # also draw a label
            font = pygame.font.Font(None, 20)
            ladder_text = font.render(f"{i}", True, (255, 255, 255))
            self.screen.blit(
                ladder_text,
                (
                    int(pos[0] + 1) * RENDER_SCALE_FACTOR,
                    int(pos[1] + 1) * RENDER_SCALE_FACTOR,
                ),
            )

        # Draw platforms
        for i in range(level_constants.platform_positions.shape[0]):
            pos = level_constants.platform_positions[i]
            # Only draw valid platforms
            if pos[0] != -1:
                pygame.draw.rect(
                    self.screen,
                    PLATFORM_COLOR,
                    (
                        int(pos[0]) * RENDER_SCALE_FACTOR,
                        int(pos[1]) * RENDER_SCALE_FACTOR,
                        int(level_constants.platform_sizes[i, 0]) * RENDER_SCALE_FACTOR,
                        int(level_constants.platform_sizes[i, 1]) * RENDER_SCALE_FACTOR,
                    ),
                )
                # label platforms
                font = pygame.font.Font(None, 20)
                plat_text = font.render(f"{i}", True, (255, 255, 255))
                self.screen.blit(
                    plat_text,
                    (
                        int(pos[0] + 1) * RENDER_SCALE_FACTOR,
                        int(pos[1] + 1) * RENDER_SCALE_FACTOR,
                    ),
                )

        # Draw fruits
        for i in range(state.level.fruit_actives.shape[0]):
            if state.level.fruit_actives[i]:
                pygame.draw.rect(
                    self.screen,
                    FRUIT_COLOR[state.level.fruit_stages[i]],
                    (
                        int(state.level.fruit_positions[i, 0]) * RENDER_SCALE_FACTOR,
                        int(state.level.fruit_positions[i, 1]) * RENDER_SCALE_FACTOR,
                        int(FRUIT_WIDTH) * RENDER_SCALE_FACTOR,
                        int(FRUIT_HEIGHT) * RENDER_SCALE_FACTOR,
                    ),
                )

        # Draw Bell
        pygame.draw.rect(
            self.screen,
            BELL_COLOR,
            (
                int(state.level.bell_position[0]) * RENDER_SCALE_FACTOR,
                int(state.level.bell_position[1]) * RENDER_SCALE_FACTOR,
                int(BELL_WIDTH) * RENDER_SCALE_FACTOR,
                int(BELL_HEIGHT) * RENDER_SCALE_FACTOR,
            ),
        )

        # Draw player
        pygame.draw.rect(
            self.screen,
            PLAYER_COLOR,
            (
                int(state.player.x) * RENDER_SCALE_FACTOR,
                int(state.player.y) * RENDER_SCALE_FACTOR,
                int(PLAYER_WIDTH) * RENDER_SCALE_FACTOR,
                int(state.player.height) * RENDER_SCALE_FACTOR,
            ),
        )

        # Draw Child
        pygame.draw.rect(
            self.screen,
            PLAYER_COLOR,
            (
                int(state.level.child_position[0]) * RENDER_SCALE_FACTOR,
                int(state.level.child_position[1]) * RENDER_SCALE_FACTOR,
                int(CHILD_WIDTH) * RENDER_SCALE_FACTOR,
                int(CHILD_HEIGHT) * RENDER_SCALE_FACTOR,
            ),
        )

        # Draw player punch effects
        if state.player.punch_left:
            pygame.draw.rect(
                self.screen,
                PLAYER_COLOR,
                (
                    int(state.player.x - 2) * RENDER_SCALE_FACTOR,
                    (int(state.player.y) + 8) * RENDER_SCALE_FACTOR,
                    2 * RENDER_SCALE_FACTOR,
                    4 * RENDER_SCALE_FACTOR,
                ),
            )

        if state.player.punch_right:
            pygame.draw.rect(
                self.screen,
                PLAYER_COLOR,
                (
                    int(state.player.x + PLAYER_WIDTH) * RENDER_SCALE_FACTOR,
                    (int(state.player.y) + 8) * RENDER_SCALE_FACTOR,
                    2 * RENDER_SCALE_FACTOR,
                    4 * RENDER_SCALE_FACTOR,
                ),
            )

        # Drawing falling coconut
        if state.level.falling_coco_position[1] != -1:
            pygame.draw.rect(
                self.screen,
                COCONUT_COLOR,
                (
                    int(state.level.falling_coco_position[0]) * RENDER_SCALE_FACTOR,
                    int(state.level.falling_coco_position[1]) * RENDER_SCALE_FACTOR,
                    int(COCONUT_WIDTH) * RENDER_SCALE_FACTOR,
                    int(COCONUT_HEIGHT) * RENDER_SCALE_FACTOR,
                ),
            )

        # Draw UI
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {state.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        orient_text = font.render(
            f"Orientation: {'left (-1)' if state.player.orientation < 0 else 'right (1)'}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(orient_text, (10, 45))

        timer_text = font.render(f"Timer: {state.level.timer}", True, (255, 255, 255))
        self.screen.blit(
            timer_text, (60 * RENDER_SCALE_FACTOR, 192 * RENDER_SCALE_FACTOR)
        )

        lives_text = font.render(f"Lives: {state.lives}", True, (255, 255, 255))
        self.screen.blit(
            lives_text, (30 * RENDER_SCALE_FACTOR, 192 * RENDER_SCALE_FACTOR)
        )

        pygame.display.flip()


if __name__ == "__main__":
    pygame.init()
    game = Game()
    renderer = Renderer()
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)
    curr_state = jitted_reset()
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
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
                        curr_state = jitted_step(curr_state, action)

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_state = jitted_step(curr_state, action)

        renderer.render(curr_state)
        counter += 1
        pygame.time.Clock().tick(90)

    pygame.quit()
