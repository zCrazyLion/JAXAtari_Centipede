from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame

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


class Fruit(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array
    active: chex.Array
    stage: chex.Array


class Level(NamedTuple):
    id: chex.Array
    ladders: chex.Array
    platforms: chex.Array
    fruits: chex.Array


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


# -------- Entity Inits --------
P_HEIGHT = 4

L1P1 = Entity(x=16, y=172, w=128, h=P_HEIGHT)
L1P2 = Entity(x=16, y=124, w=128, h=P_HEIGHT)
L1P3 = Entity(x=16, y=76, w=128, h=P_HEIGHT)
L1P4 = Entity(x=16, y=28, w=128, h=P_HEIGHT)

LADDER_HEIGHT = 35
LADDER_WIDTH = 8

L1L1 = Entity(x=132, y=132, w=LADDER_WIDTH, h=LADDER_HEIGHT)
L1L2 = Entity(x=20, y=84, w=LADDER_WIDTH, h=LADDER_HEIGHT)
L1L3 = Entity(x=132, y=36, w=LADDER_WIDTH, h=LADDER_HEIGHT)

L1F1 = Fruit(x=50, y=100, w=FRUIT_SIZE, h=FRUIT_SIZE, active=True, stage=1)
L1F2 = Fruit(x=70, y=60, w=FRUIT_SIZE, h=FRUIT_SIZE, active=True, stage=1)
L1F3 = Fruit(x=90, y=140, w=FRUIT_SIZE, h=FRUIT_SIZE, active=True, stage=1)

levels = [
    Level(
        id=1,
        ladders=[L1L1, L1L2, L1L3],
        platforms=[L1P1, L1P2, L1P3, L1P4],
        fruits=[L1F1, L1F2, L1F3],
    )
]
# -------- Game State --------
class State(NamedTuple):
    player: Player
    score: chex.Array
    fruits: chex.Array
    player_lives: chex.Array
    current_level: chex.Array
    step_counter: chex.Array


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


# -------- Functions for Clipping / Clamping and Platforms --------
def get_player_platform(state: State) -> Tuple[chex.Array]:
    """
    Returns array of booleans indicating if player is on a platform.
    """
    player_y = state.player.y
    ph = state.player.height

    on_p1 = jnp.logical_and(player_y <= (L1P1.y - ph), player_y > (L1P2.y - ph))
    on_p2 = jnp.logical_and(player_y <= (L1P2.y - ph), player_y > (L1P3.y - ph))
    on_p3 = jnp.logical_and(player_y <= (L1P3.y - ph), player_y > (L1P4.y - ph))
    on_p4 = player_y <= (L1P4.y - ph)

    return jnp.array([on_p1, on_p2, on_p3, on_p4])


def player_on_ground(state: State) -> Tuple[chex.Array]:
    """
    Returns array of booleans indicating if player is on the ground.
    """
    player_y = state.player.y
    ph = state.player.height

    on_p1, on_p2, on_p3, on_p4 = get_player_platform(state)

    on_p1_ground = jnp.where(on_p1, player_y == (L1P1.y - ph), False)
    on_p2_ground = jnp.where(on_p2, player_y == (L1P2.y - ph), False)
    on_p3_ground = jnp.where(on_p3, player_y == (L1P3.y - ph), False)
    on_p4_ground = jnp.where(on_p4, player_y == (L1P4.y - ph), False)

    return jnp.array([on_p1_ground, on_p2_ground, on_p3_ground, on_p4_ground])


# -------- Collision with entities --------
def do_collide_with_threshold(
    e1_x: chex.Array,
    e1_y: chex.Array,
    e1_w: chex.Array,
    e1_h: chex.Array,
    e2_x: chex.Array,
    e2_y: chex.Array,
    e2_w: chex.Array,
    e2_h: chex.Array,
    threshold: float,
) -> Tuple[chex.Array]:
    """
    Returns True if the two rectangles overlap by at least the threshold, False otherwise.
    The threshold is a fraction of the area of the first rectangle ranging between (0, 1].
    """

    # assertions
    chex.assert_tree_all_finite(
        jnp.array([e1_x, e1_y, e1_w, e1_h, e2_x, e2_y, e2_w, e2_h])
    )
    chex.assert_scalar_in(threshold, 0, 1)

    # Find the boundaries of the overlapping region
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


def do_collide(
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
    return do_collide_with_threshold(e1_x, e1_y, e1_w, e1_h, e2_x, e2_y, e2_w, e2_h, 0)


def virtual_hitbox_collision(
    state: State,
    entity: Entity,
    threshold: float = 0.3,
    virtual_hitbox_height: float = 12.0,
) -> chex.Array:
    return do_collide_with_threshold(
        e1_x=state.player.x,
        e1_y=state.player.y + entity.h + 1,
        e1_w=PLAYER_WIDTH,
        e1_h=virtual_hitbox_height,
        e2_x=entity.x,
        e2_y=entity.y,
        e2_w=entity.w,
        e2_h=entity.h,
        threshold=threshold,
    )


def player_is_on_ladder(
    state: State, ladder: Entity, threshold: float = 0.3
) -> chex.Array:
    """
    Checks the collision of the player with a ladder. <threshold>% of the players surface area have to overlap with the ladder.
    """
    return do_collide_with_threshold(
        e1_x=state.player.x,
        e1_y=state.player.y,
        e1_w=PLAYER_WIDTH,
        e1_h=state.player.height,
        e2_x=ladder.x,
        e2_y=ladder.y,
        e2_w=ladder.w,
        e2_h=ladder.h,
        threshold=threshold,
    )


# -------- Jumping and Climbing --------
def player_jump_controller(
    state: State, jump_pressed: chex.Array, ladder_intersect: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
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

    jump_counter = jnp.where(jump_start, 0, jump_counter)
    jump_orientation = jnp.where(
        jump_start, state.player.orientation, state.player.jump_orientation
    )
    jump_base_y = jnp.where(jump_start, player_y, jump_base_y)
    is_jumping = is_jumping | jump_start

    jump_counter = jnp.where(is_jumping, jump_counter + 1, jump_counter)

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

    jump_complete = jump_counter >= 41
    is_jumping = jnp.where(jump_complete, False, is_jumping)
    jump_counter = jnp.where(jump_complete, 0, jump_counter)

    return new_y, jump_counter, is_jumping, jump_base_y, jump_orientation


def player_climb_controller(
    state: State,
    y: chex.Array,
    press_up: chex.Array,
    press_down: chex.Array,
    ladder_intersect: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:

    # Ladder Below Collision
    collide_l1_below = virtual_hitbox_collision(state, L1L1)
    collide_l2_below = virtual_hitbox_collision(state, L1L2)
    collide_l3_below = virtual_hitbox_collision(state, L1L3)
    ladder_intersect_below = collide_l1_below | collide_l2_below | collide_l3_below

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

    def get_next_platform_below_player():
        on1, on2, on3, on4 = get_player_platform(state)
        return jnp.where(
            on2,
            L1P1.y,
            jnp.where(
                on3,
                L1P2.y,
                jnp.where(on4, L1P3.y, 0),
            ),
        )

    climb_base_y = jnp.where(
        climb_start_downward,
        get_next_platform_below_player() - state.player.height,
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

    # jax.debug.print(
    #     "isclimbing={c}, counter={co}, climb_start={cs}, climb_base={y}, climb_up={u}, climb_down={d}, climb_stop={s}, ladder_intersect={li}",
    #     c=is_climbing,
    #     co=climb_counter,
    #     cs=climb_start,
    #     y=climb_base_y,
    #     u=climb_up,
    #     d=climb_down,
    #     s=climb_stop,
    #     li=ladder_intersect,
    # )

    return new_y, is_climbing, climb_base_y, climb_counter, cooldown_counter

# -------- Player Height --------
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


# -------- Handle Fruits --------
@jax.jit
def fruits_step(state:State):

    f1 = state.fruits[0]
    f2 = state.fruits[1]
    f3 = state.fruits[2]
    
    score_addition = 0

    # Fruit at Slot 1
    fruit1_collision = do_collide(state.player.x, state.player.y, PLAYER_WIDTH, state.player.height, f1.x, f1.y, f1.w, f1.h)
    f1_collision_condition = jnp.logical_and(fruit1_collision, f1.active)

    score_addition = jnp.where(f1_collision_condition, 100, score_addition)
    
    new_active = jnp.where(f1_collision_condition, False, f1.active)
    new_f1 = Fruit(x=f1.x, y=f1.y, w=f1.w, h=f1.h, active=new_active, stage=f1.stage)

    # Fruit at Slot 2
    fruit2_collision = do_collide(state.player.x, state.player.y, PLAYER_WIDTH, state.player.height, f2.x, f2.y, f2.w, f2.h)
    f2_collision_condition = jnp.logical_and(fruit2_collision, f2.active)

    score_addition = jnp.where(f2_collision_condition, 100, score_addition)
    
    new_active = jnp.where(f2_collision_condition, False, f2.active)
    new_f2 = Fruit(x=f2.x, y=f2.y, w=f2.w, h=f2.h, active=new_active, stage=f2.stage)

    # Fruit at Slot 3
    fruit3_collision = do_collide(state.player.x, state.player.y, PLAYER_WIDTH, state.player.height, f3.x, f3.y, f3.w, f3.h)
    f3_collision_condition = jnp.logical_and(fruit3_collision, f3.active)

    score_addition = jnp.where(f3_collision_condition, 100, score_addition)
    
    new_active = jnp.where(f3_collision_condition, False, f3.active)
    new_f3 = Fruit(x=f3.x, y=f3.y, w=f3.w, h=f3.h, active=new_active, stage=f3.stage)

    # Return Values
    new_fruits = [new_f1, new_f2, new_f3]
    return score_addition, new_fruits 

# -------- Main Function for Player Movement --------
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
]:
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

    # Ladder Collision
    # Hardcoded Approach
    collide_l1_thresh = player_is_on_ladder(state, L1L1)
    collide_l2_thresh = player_is_on_ladder(state, L1L2)
    collide_l3_thresh = player_is_on_ladder(state, L1L3)
    ladder_intersect_thresh = collide_l1_thresh | collide_l2_thresh | collide_l3_thresh

    collide_l1 = player_is_on_ladder(state, L1L1, 0)
    collide_l2 = player_is_on_ladder(state, L1L2, 0)
    collide_l3 = player_is_on_ladder(state, L1L3, 0)
    ladder_intersect_no_thresh = collide_l1 | collide_l2 | collide_l3

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
    on_p1, on_p2, on_p3, on_p4 = get_player_platform(state)

    y = jnp.where(
        on_p1,
        jnp.where(
            ~state.player.is_climbing & new_is_climbing & press_down,
            y,
            jnp.clip(y, 0, L1P1.y - new_player_height),
        ),
        y,
    )
    y = jnp.where(
        on_p2,
        jnp.where(
            ~state.player.is_climbing & new_is_climbing & press_down,
            y,
            jnp.clip(y, 0, L1P2.y - new_player_height),
        ),
        y,
    )
    y = jnp.where(
        on_p3,
        jnp.where(
            ~state.player.is_climbing & new_is_climbing & press_down,
            y,
            jnp.clip(y, 0, L1P3.y - new_player_height),
        ),
        y,
    )
    y = jnp.where(
        on_p4,
        jnp.where(
            ~state.player.is_climbing & new_is_climbing & press_down,
            y,
            jnp.clip(y, 0, L1P4.y - new_player_height),
        ),
        y,
    )

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
                x=PLAYER_START_X,
                y=PLAYER_START_Y,
                vel_x=0,
                is_crouching=False,
                is_jumping=False,
                is_climbing=False,
                jump_counter=0,
                orientation=1,
                jump_base_y=PLAYER_START_Y,
                height=PLAYER_HEIGHT,
                jump_orientation=0,
                climb_base_y=PLAYER_START_Y,
                climb_counter=0,
                punch_left=False,
                punch_right=False,
                cooldown_counter=0,
            ),
            score=0,
            fruits = [L1F1, L1F2, L1F3],
            player_lives=3,
            current_level=1,
            step_counter=0,
        )

    @chex.chexify
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        reset_cond = jnp.any(jnp.array([action == RESET]))

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

        (
            fruit_score,
            new_fruits
        ) = fruits_step(state)

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
                score=state.score + fruit_score,
                fruits=new_fruits,
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

        # Draw level objects
        level = next((l for l in levels if l.id == state.current_level), None)
        if level is not None:
            for ladder in level.ladders:
                pygame.draw.rect(
                    self.screen,
                    LADDER_COLOR,
                    (
                        ladder.x * RENDER_SCALE_FACTOR,
                        ladder.y * RENDER_SCALE_FACTOR,
                        ladder.w * RENDER_SCALE_FACTOR,
                        ladder.h * RENDER_SCALE_FACTOR,
                    ),
                )
            for platform in level.platforms:
                pygame.draw.rect(
                    self.screen,
                    PLATFORM_COLOR,
                    (
                        platform.x * RENDER_SCALE_FACTOR,
                        platform.y * RENDER_SCALE_FACTOR,
                        platform.w * RENDER_SCALE_FACTOR,
                        platform.h * RENDER_SCALE_FACTOR,
                    ),
                )
            for fruit in state.fruits:
                if fruit.active:
                    pygame.draw.rect(
                        self.screen,
                        FRUIT_COLOR,
                        (
                            fruit.x * RENDER_SCALE_FACTOR,
                            fruit.y * RENDER_SCALE_FACTOR,
                            fruit.w * RENDER_SCALE_FACTOR,
                            fruit.h * RENDER_SCALE_FACTOR,
                        ),
                    )

        # Draw player
        pygame.draw.rect(
            self.screen,
            PLAYER_COLOR,
            (
                int(state.player.x) * RENDER_SCALE_FACTOR,
                int(state.player.y) * RENDER_SCALE_FACTOR,
                PLAYER_WIDTH * RENDER_SCALE_FACTOR,
                state.player.height * RENDER_SCALE_FACTOR,
            ),
        )

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

        # Draw score
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
    jitted_step = game.step
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
