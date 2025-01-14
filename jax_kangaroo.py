from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
from jax import Array

from kangaroo_levels import Kangaroo_Level_1, Kangaroo_Level_2, Kangaroo_Level_3

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
RENDER_SCALE_FACTOR = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210
PLAYER_WIDTH, PLAYER_HEIGHT = 8, 24
ENEMY_WIDTH, ENEMY_HEIGHT = 8, 24
FRUIT_SIZE = 8

BACKGROUND_COLOR = (66, 72, 200)
PLAYER_COLOR = (255, 145, 0)
ENEMY_COLOR = (236, 200, 96)
FRUIT_COLOR = (255, 0, 0)
PLATFORM_COLOR = (130, 74, 0)
LADDER_COLOR = (199, 148, 97)

PLAYER_START_X, PLAYER_START_Y = 23, 148
MOVEMENT_SPEED = 1

LEFT_CLIP = 16
RIGHT_CLIP = 144


# -------- Entity Classes --------
class Entity(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array


class Player(NamedTuple):
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    is_crouching: chex.Array
    is_jumping: chex.Array
    is_climbing: chex.Array
    jump_counter: chex.Array
    orientation: chex.Array
    jump_base_y: chex.Array
    height: chex.Array
    jump_orientation: chex.Array
    climb_base_y: chex.Array
    climb_counter: chex.Array
    punch_left: chex.Array
    punch_right: chex.Array
    cooldown_counter: chex.Array


class State(NamedTuple):
    player: Player
    score: chex.Array
    fruit_positions_x: chex.Array
    fruit_positions_y: chex.Array
    fruit_actives: chex.Array
    fruit_stages: chex.Array
    player_lives: chex.Array
    current_level: chex.Array
    step_counter: chex.Array


# Level Constants
LADDER_HEIGHT = jnp.array(35)
LADDER_WIDTH = jnp.array(8)
P_HEIGHT = jnp.array(4)


class LevelConstants(NamedTuple):
    ladder_positions: chex.Array  # shape (num_ladders, 2) for x,y positions
    ladder_sizes: chex.Array  # shape (num_ladders, 2) for width,height
    platform_positions: chex.Array  # shape (num_platforms, 2) for x,y positions
    platform_sizes: chex.Array  # shape (num_platforms, 2) for width,height


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
def get_player_platform(state: State, level_constants: LevelConstants) -> chex.Array:
    """Returns array of booleans indicating if player is on a platform."""
    player_x = state.player.x
    player_y = state.player.y
    ph = state.player.height
    pw = PLAYER_WIDTH

    platform_positions = level_constants.platform_positions  # [N, 2] array of (x, y)
    platform_sizes = level_constants.platform_sizes  # [N, 2] array of (width, height)

    def calculate_upper_platform_index(i, carry):
        # Unpack the carry tuple
        prev_lowest_diff, best_index = carry

        platform_x = platform_positions[i, 0]
        platform_width = platform_sizes[i, 0]
        platform_y = platform_positions[i, 1]

        is_within_x = jnp.logical_and(
            player_x + pw > platform_x,
            player_x < platform_x + platform_width
        )

        is_above_player = player_y > platform_y

        diff = jnp.where(is_above_player, player_y - platform_y, jnp.inf)

        # Update both the lowest difference and the corresponding index
        new_diff = jnp.where(
            jnp.logical_and(is_within_x, is_above_player),
            jnp.minimum(prev_lowest_diff, diff),
            prev_lowest_diff
        )

        new_index = jnp.where(
            jnp.logical_and(is_within_x, jnp.logical_and(is_above_player, diff < prev_lowest_diff)),
            i,
            best_index
        )

        return new_diff, new_index

    # Initialize with (difference, index)
    _, platform_over_player_index = jax.lax.fori_loop(
        0,
        platform_positions.shape[0],
        calculate_upper_platform_index,
        (jnp.inf, -1)
    )

    def calculate_lower_platform_index(i, carry):
        # Unpack the carry tuple
        prev_lowest_diff, best_index = carry

        platform_x = platform_positions[i, 0]
        platform_width = platform_sizes[i, 0]
        platform_y = platform_positions[i, 1]

        is_within_x = jnp.logical_and(
            player_x + pw > platform_x,
            player_x < platform_x + platform_width
        )

        is_below_player = player_y < platform_y

        diff = jnp.where(is_below_player, platform_y - player_y, jnp.inf)

        # Update both the lowest difference and the corresponding index
        new_diff = jnp.where(
            jnp.logical_and(is_within_x, is_below_player),
            jnp.minimum(prev_lowest_diff, diff),
            prev_lowest_diff
        )

        new_index = jnp.where(
            jnp.logical_and(is_within_x, jnp.logical_and(is_below_player, diff < prev_lowest_diff)),
            i,
            best_index
        )

        return new_diff, new_index

    # Initialize with (difference, index)
    _, platform_under_player_index = jax.lax.fori_loop(
        0,
        platform_positions.shape[0],
        calculate_lower_platform_index,
        (jnp.inf, -1)
    )

    def check_platform(i, platform_bands):
        platform_x = platform_positions[i, 0]
        platform_width = platform_sizes[i, 0]

        lower_platform_y = platform_positions[i, 1]

        is_between_platforms = jnp.logical_and(
            # TODO: the 0 does not belong here, but this function should be replaced anyway
            player_y <= (lower_platform_y - ph), player_y > (0 - ph)
        )

        # X-axis overlap check
        is_within_x = jnp.logical_and(
            player_x + pw > platform_x,
            player_x < platform_x + platform_width
        )

        # Combine x and y checks
        is_on_platform = jnp.logical_and(is_within_x, is_between_platforms)

        is_valid = is_valid_platform(platform_positions[i])

        return platform_bands.at[i].set(is_on_platform & is_valid)

    print("upper", platform_over_player_index)
    print("lower", platform_under_player_index)

    initial_bands = jnp.zeros(platform_positions.shape[0], dtype=bool)

    # TODO: this is legacy, use the new method with the calculated lower and upper platforms
    return jax.lax.fori_loop(0, platform_positions.shape[0], check_platform, initial_bands)


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
    """Returns True if rectangles overlap by at least threshold fraction."""
    overlap_start_x = jnp.maximum(e1_x, e2_x)
    overlap_end_x = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
    overlap_start_y = jnp.maximum(e1_y, e2_y)
    overlap_end_y = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

    # Calculate dimensions of overlap region
    overlap_width = overlap_end_x - overlap_start_x
    overlap_height = overlap_end_y - overlap_start_y

    # Calculate area of overlap
    overlap_area = overlap_width * overlap_height

    # Calculate area of first rectangle
    rect1_area = e1_w * e1_h

    # Calculate minimum required overlap area based on threshold
    min_required_overlap = rect1_area * threshold

    # Check if overlap exceeds required threshold
    meets_threshold = overlap_area >= min_required_overlap

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
    state: State,
    level_constants: LevelConstants,
    threshold: float = 0.3,
    virtual_hitbox_height: float = 12.0,
) -> chex.Array:
    """Checks collision between a virtual hitbox below player and ladders."""

    def check_single_collision(i, collisions):
        ladder_pos = level_constants.ladder_positions[i]
        ladder_size = level_constants.ladder_sizes[i]

        collision = entities_collide_with_threshold(
            state.player.x,
            state.player.y + ladder_size[1] + 1,
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
    state: State, level_constants: LevelConstants, threshold: float = 0.3
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
    state: State,
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
    state: State, jump_pressed: chex.Array, ladder_intersect: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
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
    jump_base_y = state.player.jump_base_y

    cooldown_condition = state.player.cooldown_counter > 0
    jump_start = jump_pressed & ~is_jumping & ~ladder_intersect & ~cooldown_condition

    # Update jump state on start
    jump_counter = jnp.where(jump_start, 0, jump_counter)
    jump_orientation = jnp.where(
        jump_start, state.player.orientation, state.player.jump_orientation
    )
    jump_base_y = jnp.where(jump_start, player_y, jump_base_y)
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

    total_offset = offset_for(jump_counter)
    new_y = jnp.where(is_jumping, jump_base_y + total_offset, player_y)

    # Check for jump completion
    jump_complete = jump_counter >= 41
    is_jumping = jnp.where(jump_complete, False, is_jumping)
    jump_counter = jnp.where(jump_complete, 0, jump_counter)

    return new_y, jump_counter, is_jumping, jump_base_y, jump_orientation


@partial(jax.jit, static_argnums=())
def player_climb_controller(
    state: State,
    y: chex.Array,
    press_up: chex.Array,
    press_down: chex.Array,
    ladder_intersect: chex.Array,
) -> tuple[Array, Array, Array, Array, Array]:

    # Ladder Below Collision
    level_constants = get_level_constants(state.current_level)
    ladder_intersect_below = jnp.any(player_is_above_ladder(state, level_constants))

    new_y = y
    is_climbing = state.player.is_climbing
    is_climbing = jnp.where(state.player.is_jumping, False, is_climbing)

    climb_counter = state.player.climb_counter

    climb_start = press_up & ~is_climbing & ladder_intersect & ~state.player.is_jumping
    climb_start_downward = (
        press_down & ~is_climbing & ladder_intersect_below & ~state.player.is_jumping
    )

    is_climbing = is_climbing | climb_start | climb_start_downward

    climb_counter = jnp.where(climb_start | climb_start_downward, 0, climb_counter)

    climb_base_y = state.player.climb_base_y
    climb_base_y = jnp.where(climb_start, new_y, state.player.climb_base_y)

    climb_base_y = jnp.where(
        climb_start_downward,
        get_next_platform_below_player(state, level_constants) - state.player.height,
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

    # Check if not climbing anymore -> bottom of ladder
    climb_stop = jnp.logical_and(is_climbing, jnp.greater_equal(new_y, climb_base_y))

    is_climbing = jnp.where(climb_stop, False, is_climbing)

    # Check if not climbing anymore -> top of ladder
    is_climbing = jnp.where(ladder_intersect | climb_start_downward, is_climbing, False)

    clock_reset = climb_counter >= 19
    climb_counter = jnp.where(clock_reset, 0, climb_counter)
    cooldown_counter = jnp.where(
        clock_reset,
        15,
        state.player.cooldown_counter - 1,
    )

    return new_y, is_climbing, climb_base_y, climb_counter, cooldown_counter


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
def get_next_platform_below_player(
    state: State, level_constants: LevelConstants
) -> chex.Array:
    """Gets the y-position of the next platform below the player."""
    platform_bands = get_player_platform(state, level_constants)
    platform_ys = level_constants.platform_positions[:, 1]

    def find_next_platform(i, current_platform_y):
        is_valid = is_valid_platform(level_constants.platform_positions[i])
        is_in_band = jnp.logical_and(platform_bands[i], is_valid)
        return jnp.where(is_in_band, platform_ys[i - 1], current_platform_y)

    initial_y = platform_ys[0]  # Start with platform at 0
    return jax.lax.fori_loop(0, platform_ys.shape[0], find_next_platform, initial_y)


@partial(jax.jit, static_argnums=())
def fruits_step(state: State) -> Tuple[chex.Array, chex.Array]:
    """Handles fruit collection and scoring."""

    def check_fruit(i, carry):
        score, actives = carry

        fruit_collision = entities_collide(
            state.player.x,
            state.player.y,
            PLAYER_WIDTH,
            state.player.height,
            state.fruit_positions_x[i],
            state.fruit_positions_y[i],
            FRUIT_SIZE,
            FRUIT_SIZE,
        )

        collision_condition = jnp.logical_and(fruit_collision, actives[i])
        new_score = jnp.where(collision_condition, score + 100, score)
        new_actives = actives.at[i].set(
            jnp.where(collision_condition, False, actives[i])
        )

        return new_score, new_actives

    initial_score = jnp.array(0)
    initial_actives = state.fruit_actives

    return jax.lax.fori_loop(
        0, len(state.fruit_actives), check_fruit, (initial_score, initial_actives)
    )


def pad_array(arr, target_size):
    """Pads a 2D array with -1s to reach target size in first dimension."""
    current_size = arr.shape[0]
    if current_size >= target_size:
        return arr

    pad_size = target_size - current_size
    return jnp.pad(arr, ((0, pad_size), (0, 0)), mode='constant', constant_values=-1)


def pad_to_size(level_constants, max_platforms):
    """Pads all arrays in level constants to specified size."""
    return LevelConstants(
        ladder_positions=pad_array(level_constants.ladder_positions, max_platforms),
        ladder_sizes=pad_array(level_constants.ladder_sizes, max_platforms),
        platform_positions=pad_array(level_constants.platform_positions, max_platforms),
        platform_sizes=pad_array(level_constants.platform_sizes, max_platforms)
    )

@partial(jax.jit, static_argnums=())
def get_level_constants(current_level):
    """Returns constants for the current level."""
    # TODO: this is necessary due to JAX JIT compatibility (it forces the same length for all arrays). Fun isn't it?
    max_platforms = 20  # Maximum across all levels

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
            operand=None
        ),
        operand=None
    )

@partial(jax.jit, static_argnums=())
def player_step(state: State, action: chex.Array) -> Tuple[
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
    chex.Array,
]:
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

    # TODO: Change inputs based on state or better keep inputs but set local constants based on state for use in controllers
    # e.g. instead of:
    # press_down = jnp.where(state.is_jumping, False, press_down)
    # use:
    # IS_JUMPING = state.is_jumping
    # in the fire/climbing controller: if IS_JUMPING -> do nothing
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
    is_punching_left = jnp.logical_and(press_fire, is_looking_left)
    is_punching_right = jnp.logical_and(press_fire, is_looking_right)

    # check for any collision with standard threshold
    ladder_intersect_thresh = jnp.any(check_ladder_collisions(state, level_constants))

    ladder_intersect_no_thresh = jnp.any(
        check_ladder_collisions(state, level_constants, 0)
    )

    ladder_intersect = jnp.where(
        state.player.is_climbing, ladder_intersect_no_thresh, ladder_intersect_thresh
    )

    # Jump controller
    new_y, new_jump_counter, new_is_jumping, new_jump_base_y, new_jump_orientation = (
        player_jump_controller(state, press_up, ladder_intersect)
    )

    # Climb controller
    (
        new_y,
        new_is_climbing,
        new_climb_base_y,
        new_climb_counter,
        new_cooldown_counter,
    ) = player_climb_controller(state, new_y, press_up, press_down, ladder_intersect)

    new_is_crouching = jnp.logical_and(
        press_down, jnp.logical_and(~new_is_climbing, ~new_is_jumping)
    )

    # Calculate horizontal velocity
    candidate_vel_x = jnp.where(
        new_is_crouching,
        0,
        jnp.where(
            press_left, -MOVEMENT_SPEED, jnp.where(press_right, MOVEMENT_SPEED, 0)
        ),
    )

    # Check Orientation (Left/Right)
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
    # If height changes, shift the player's top so the bottom remains consistent
    dy = old_height - new_player_height
    y = new_y + dy

    # x-axis movement
    x = jnp.clip(x + vel_x, LEFT_CLIP, RIGHT_CLIP - PLAYER_WIDTH)

    # y-axis movement
    platform_bools = get_player_platform(state, level_constants)
    platform_ys = level_constants.platform_positions[:, 1]

    def get_platform_dependent_y(i, curr_y):
        is_valid = is_valid_platform(level_constants.platform_positions[i])
        platform_y = jnp.where(
            jnp.logical_and(platform_bools[i], is_valid),
            jnp.where(
                ~state.player.is_climbing & new_is_climbing & press_down,
                y,
                jnp.clip(y, 0, platform_ys[i] - new_player_height),
            ),
            curr_y,
        )
        return platform_y

    # iterate the platform bools and check which is true, then return the corresponding y value
    y = jax.lax.fori_loop(0, len(platform_bools), get_platform_dependent_y, y)

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
        new_player_height,
        new_jump_orientation,
        new_climb_base_y,
        new_climb_counter,
        is_punching_left,
        is_punching_right,
        new_cooldown_counter,
    )


# -------- Game Interface for Reset and Step --------
class Game:
    def __init__(self, frameskip: int = 1):
        self.frameskip = frameskip

    def reset(self) -> State:
        return State(
            player=Player(
                x=jnp.array(PLAYER_START_X),
                y=jnp.array(PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(PLAYER_START_Y),
                height=jnp.array(PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
            ),
            score=jnp.array(0),
            # TODO: pull these in the levels as well, right?
            # Answer: yes but we have to put the whole level in the state (so the agent knows where the platforms and the ladders etc are)
            fruit_positions_x=jnp.array([119, 39, 59]),
            fruit_positions_y=jnp.array([108, 84, 60]),
            fruit_actives=jnp.ones(3, dtype=jnp.bool_),
            fruit_stages=jnp.ones(3, dtype=jnp.int32),
            player_lives=jnp.array(3),
            current_level=jnp.array(2),
            step_counter=jnp.array(0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
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
            new_player_height,
            new_jump_orientation,
            climb_base_y,
            climb_counter,
            punch_left,
            punch_right,
            cooldown_counter,
        ) = player_step(state, action)

        # Handle fruit collection
        score_addition, new_actives = fruits_step(state)

        return jax.lax.cond(
            reset_cond,
            lambda: self.reset(),
            lambda: State(
                player=Player(
                    x=player_x,
                    y=player_y,
                    vel_x=vel_x,
                    is_crouching=is_crouching,
                    is_jumping=is_jumping,
                    is_climbing=is_climbing,
                    jump_counter=jump_counter,
                    orientation=orientation,
                    jump_base_y=jump_base_y,
                    height=new_player_height,
                    jump_orientation=new_jump_orientation,
                    climb_base_y=climb_base_y,
                    climb_counter=climb_counter,
                    punch_left=punch_left,
                    punch_right=punch_right,
                    cooldown_counter=cooldown_counter,
                ),
                score=state.score + score_addition,
                fruit_actives=new_actives,
                fruit_positions_x=state.fruit_positions_x,
                fruit_positions_y=state.fruit_positions_y,
                fruit_stages=state.fruit_stages,
                player_lives=state.player_lives,
                current_level=state.current_level,
                step_counter=state.step_counter + 1,
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

    def render(self, state: State):
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

        # Draw fruits
        for i in range(len(state.fruit_actives)):
            if state.fruit_actives[i]:
                pygame.draw.rect(
                    self.screen,
                    FRUIT_COLOR,
                    (
                        int(state.fruit_positions_x[i]) * RENDER_SCALE_FACTOR,
                        int(state.fruit_positions_y[i]) * RENDER_SCALE_FACTOR,
                        int(FRUIT_SIZE) * RENDER_SCALE_FACTOR,
                        int(FRUIT_SIZE) * RENDER_SCALE_FACTOR,
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
        pygame.time.Clock().tick(60)

    pygame.quit()
