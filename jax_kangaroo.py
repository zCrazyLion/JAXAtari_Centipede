from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import numpy as np
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

# -------- Game constants --------
RENDER_SCALE_FACTOR = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210
PLAYER_WIDTH, PLAYER_HEIGHT = 8, 24
ENEMY_WIDTH, ENEMY_HEIGHT = 8, 24
FRUIT_SIZE = 8

BACKGROUND_COLOR = (66, 72, 200)
PLAYER_COLOR = (255, 145, 0)
ENEMY_COLOR = (236, 200, 96)
FRUIT_COLOR = (92, 186, 92)
PLATFORM_COLOR = (130, 74, 0)
LADDER_COLOR = (199, 148, 97)

PLAYER_START_X, PLAYER_START_Y = 23, 148
GRAVITY = 0.5
JUMP_VELOCITY = -8
MOVEMENT_SPEED = 1

LEFT_CLIP = 16
RIGHT_CLIP = 144


# -------- Entity Classes --------
class Entity(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array

# -------- Entity Inits --------
P_HEIGHT = 4

L1P1 = Entity(x=16, y=172, w=128, h=P_HEIGHT)
L1P2 = Entity(x=16, y=124, w=128, h=P_HEIGHT)
L1P3 = Entity(x=16, y=76, w=128, h=P_HEIGHT)
L1P4 = Entity(x=16, y=28, w=128, h=P_HEIGHT)

LADDER_HEIGHT = 35
LADDER_WIDTH = 8

L1L1 = Entity(x=132, y=132, w=LADDER_WIDTH, h=LADDER_HEIGHT)
L1L2 = Entity(x=20, y=85, w=LADDER_WIDTH, h=LADDER_HEIGHT)
L1L3 = Entity(x=132, y=37, w=LADDER_WIDTH, h=LADDER_HEIGHT)


# -------- Game State --------
class State(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_vel_x: chex.Array
    player_vel_y: chex.Array
    is_crouching: chex.Array
    player_score: chex.Array
    player_lives: chex.Array
    current_level: chex.Array
    is_jumping: chex.Array
    is_climbing: chex.Array
    step_counter: chex.Array
    jump_counter: chex.Array
    climb_counter: chex.Array
    falling_coco_x: chex.Array
    falling_coco_y: chex.Array
    orientation: chex.Array
    jump_base_y: chex.Array
    climb_base_y: chex.Array
    jump_orientation: chex.Array
    player_height: chex.Array
    punch_left: chex.Array
    punch_right: chex.Array


# -------- Keyboard Inputs --------
def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]

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
    Returns booleans for which platform "band" the player's top is in,
    using top-based y logic:  y <= (platform.y - height).
    """
    player_y = state.player_y
    ph = state.player_height

    on_p1 = jnp.where(
        jnp.logical_and(player_y <= (L1P1.y - ph), player_y > (L1P2.y - ph)),
        True,
        False,
    )

    on_p2 = jnp.where(
        jnp.logical_and(player_y <= (L1P2.y - ph), player_y > (L1P3.y - ph)),
        True,
        False,
    )

    on_p3 = jnp.where(
        jnp.logical_and(player_y <= (L1P3.y - ph), player_y > (L1P4.y - ph)),
        True,
        False,
    )

    on_p4 = jnp.where(
        player_y <= (L1P4.y - ph),
        True,
        False,
    )

    return jnp.array([on_p1, on_p2, on_p3, on_p4])


def player_on_ground(state: State, action: chex.Array) -> Tuple[chex.Array]:
    player_y = state.player_y

    on_p1, on_p2, on_p3, on_p4 = get_player_platform(state)

    on_p1_ground = jnp.where(
        on_p1, jnp.where(player_y == (L1P1.y - state.player_height), True, False), False
    )
    on_p2_ground = jnp.where(
        on_p2, jnp.where(player_y == (L1P2.y - state.player_height), True, False), False
    )
    on_p3_ground = jnp.where(
        on_p3, jnp.where(player_y == (L1P3.y - state.player_height), True, False), False
    )
    on_p4_ground = jnp.where(
        on_p4, jnp.where(player_y == (L1P4.y - state.player_height), True, False), False
    )

    return jnp.any(jnp.array([on_p1_ground, on_p2_ground, on_p3_ground, on_p4_ground]))


# -------- Collision with entities --------
@jax.jit
def check_collision(
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
    Returns True if the two rectangles overlap, False otherwise.
    We assume (x, y) is the top-left corner, w is width, h is height.
    """
    e1_left = e1_x
    e1_right = e1_x + e1_w
    e1_top = e1_y
    e1_bottom = e1_y + e1_h

    e2_left = e2_x
    e2_right = e2_x + e2_w
    e2_top = e2_y
    e2_bottom = e2_y + e2_h

    no_overlap = (
        (e1_bottom < e2_top)
        | (e1_top > e2_bottom)
        | (e1_right < e2_left)
        | (e1_left > e2_right)
    )

    return jnp.logical_not(no_overlap)


@jax.jit
def check_collision_with_threshold(
    e1_x: chex.Array,
    e1_y: chex.Array,
    e1_w: chex.Array,
    e1_h: chex.Array,
    e2_x: chex.Array,
    e2_y: chex.Array,
    e2_w: chex.Array,
    e2_h: chex.Array,
    threshold: float,
) -> chex.Array:

    inter_left = jnp.maximum(e1_x, e2_x)
    inter_right = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
    inter_top = jnp.maximum(e1_y, e2_y)
    inter_bottom = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

    inter_width = jnp.maximum(inter_right - inter_left, 0)
    inter_height = jnp.maximum(inter_bottom - inter_top, 0)

    intersection_area = inter_width * inter_height

    e1_area = e1_w * e1_h

    required_overlap = e1_area * threshold

    overlap_exceeds = intersection_area >= required_overlap

    return overlap_exceeds


def player_on_ladder(
    state: State, ladder: Entity, threshold: float = 0.3
) -> chex.Array:
    return check_collision_with_threshold(
        e1_x=state.player_x,
        e1_y=state.player_y,
        e1_w=PLAYER_WIDTH,
        e1_h=state.player_height,
        e2_x=ladder.x,
        e2_y=ladder.y,
        e2_w=ladder.w,
        e2_h=ladder.h,
        threshold=threshold,
    )


def player_on_ladder_no_thresh(
    state: State, ladder: Entity, threshold: float = 0.3
) -> chex.Array:
    return check_collision(
        e1_x=state.player_x,
        e1_y=state.player_y,
        e1_w=PLAYER_WIDTH,
        e1_h=state.player_height,
        e2_x=ladder.x,
        e2_y=ladder.y,
        e2_w=ladder.w,
        e2_h=ladder.h,
    )


# -------- Jumping and Climbing --------
@jax.jit
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
    player_y = state.player_y
    jump_counter = state.jump_counter
    is_jumping = state.is_jumping
    jump_base_y = state.jump_base_y

    jump_start = jump_pressed & ~is_jumping & ~ladder_intersect

    jump_counter = jnp.where(jump_start, 0, jump_counter)
    jump_orientation = jnp.where(jump_start, state.orientation, state.jump_orientation)
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

@jax.jit
def player_climb_controller(
    state: State, y:chex.Array, press_up: chex.Array, press_down: chex.Array, ladder_intersect: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:

    new_y = y
    is_climbing = state.is_climbing

    climb_counter = state.climb_counter
    
    climb_start = press_up & ~is_climbing & ladder_intersect
    is_climbing = is_climbing | climb_start

    climb_counter = jnp.where(climb_start, 0, climb_counter)
    climb_base_y = jnp.where(climb_start, new_y, state.climb_base_y)
    new_y = jnp.where(climb_start, new_y - 8, new_y)

    climb_counter = jnp.where(is_climbing, climb_counter + 1, climb_counter)
    
    climb_up = jnp.logical_and(press_up, is_climbing)
    climb_down = jnp.logical_and(press_down, is_climbing)

    new_y = jnp.where(jnp.logical_and(climb_up, jnp.equal(climb_counter, 19)), new_y - 8, new_y)
    new_y = jnp.where(jnp.logical_and(climb_down, jnp.equal(climb_counter, 19)), new_y + 8, new_y)

    climb_stop = jnp.logical_and(is_climbing,jnp.greater_equal(new_y, climb_base_y))

    is_climbing = jnp.where(climb_stop, False, is_climbing)
    is_climbing = jnp.where(ladder_intersect, is_climbing, False)

    clock_reset = climb_counter > 19
    climb_counter = jnp.where(clock_reset, 0, climb_counter)

    jax.debug.print(
        "isclimbing={c}, counter={co}, climb_start={cs}, climb_base={y}, climb_up={u}, climb_down={d}, climb_stop={s}",
        c=is_climbing, co=climb_counter, cs=climb_start, y=climb_base_y,u=climb_up, d=climb_down,s=climb_stop
    )

    return new_y, is_climbing, climb_base_y, climb_counter

# -------- Player Height --------
@jax.jit
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

# -------- Main Function for Player Movement --------
@jax.jit
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
    x, y = state.player_x, state.player_y
    old_height = state.player_height
    old_orientation = state.orientation

    # Get inputs
    press_right = jnp.any(
        jnp.array([action == RIGHT, action == UPRIGHT, action == DOWNRIGHT])
    )

    press_left = jnp.any(
        jnp.array([action == LEFT, action == UPLEFT, action == DOWNLEFT])
    )

    press_up = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT]))


    press_fire = jnp.any(
        jnp.array([action == FIRE, action == DOWNFIRE, action == UPLEFTFIRE, action == UPRIGHTFIRE, action == DOWNLEFTFIRE, action == DOWNRIGHTFIRE])
    )

    press_down_fire = jnp.any(
        jnp.array(action == DOWNFIRE)
    )

    press_down = jnp.any(
        jnp.array([action == DOWN, action == DOWNLEFT, action == DOWNRIGHT])
    )

    
    press_down = jnp.where(state.is_jumping, False, press_down)
    press_fire = jnp.where(state.is_jumping, False, press_fire)
    press_fire = jnp.where(state.is_climbing, False, press_fire)
    press_fire = jnp.where(press_down_fire, False, press_fire)

    press_up = jnp.where(press_down_fire, False, press_up)

    # Forbid left/right movement while climbing
    press_right = jnp.where(state.is_climbing, False, press_right)
    press_left = jnp.where(state.is_climbing, False, press_left)

    new_is_crouching = jnp.logical_and(press_down, jnp.logical_and(~state.is_climbing,~state.is_jumping))
    
    is_looking_left = state.orientation == -1
    is_looking_right = state.orientation == 1
    is_punching_left = jnp.logical_and(press_fire, is_looking_left)
    is_punching_right = jnp.logical_and(press_fire, is_looking_right)

    # Calculate horizontal velocity
    candidate_vel_x = jnp.where(
        new_is_crouching,
        0,
        jnp.where(
            press_left, -MOVEMENT_SPEED, jnp.where(press_right, MOVEMENT_SPEED, 0)
        ),
    )

    # Ladder Collision 
    # Hardcoded Approach
    collide_l1_thresh = player_on_ladder(state, L1L1)
    collide_l2_thresh = player_on_ladder(state, L1L2)
    collide_l3_thresh = player_on_ladder(state, L1L3)
    ladder_intersect_thresh = jnp.logical_or(collide_l1_thresh, jnp.logical_or(collide_l2_thresh, collide_l3_thresh))

    collide_l1 = player_on_ladder_no_thresh(state, L1L1)
    collide_l2 = player_on_ladder_no_thresh(state, L1L2)
    collide_l3 = player_on_ladder_no_thresh(state, L1L3)
    ladder_intersect_no_thresh = jnp.logical_or(collide_l1, jnp.logical_or(collide_l2, collide_l3))

    ladder_intersect = jnp.where(state.is_climbing, ladder_intersect_no_thresh, ladder_intersect_thresh)

    # Jump controller
    new_y, new_jump_counter, new_is_jumping, new_jump_base_y, new_jump_orientation = (
        player_jump_controller(state, press_up, ladder_intersect)
    )

    # Climb controller
    new_y, new_is_climbing, new_climb_base_y, new_climb_counter = (
       player_climb_controller(state, new_y, press_up, press_down, ladder_intersect)
    )

    # Check Orientation (Left/Right)
    standing_still = jnp.equal(candidate_vel_x, 0)
    new_orientation = jnp.sign(candidate_vel_x)
    new_orientation = jnp.where(standing_still, old_orientation, new_orientation)
    
    stop_in_air = jnp.logical_and(
        new_is_jumping, state.jump_orientation != new_orientation
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

    y = jax.lax.cond(
        on_p1, lambda y: jnp.clip(y, 0, L1P1.y - new_player_height), lambda _: _, y
    )
    y = jax.lax.cond(
        on_p2, lambda y: jnp.clip(y, 0, L1P2.y - new_player_height), lambda _: _, y
    )
    y = jax.lax.cond(
        on_p3, lambda y: jnp.clip(y, 0, L1P3.y - new_player_height), lambda _: _, y
    )
    y = jax.lax.cond(
        on_p4, lambda y: jnp.clip(y, 0, L1P4.y - new_player_height), lambda _: _, y
    )

    return (
        x,
        y,
        vel_x,
        state.player_vel_y,
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
        is_punching_right
    )


# -------- Game Interface for Reset and Step --------
class Game:
    def __init__(self, frameskip: int = 1):
        self.frameskip = frameskip

    def reset(self) -> State:
        return State(
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_vel_x=jnp.array(0),
            player_vel_y=jnp.array(0),
            is_crouching = jnp.array(False),
            player_score=jnp.array(0),
            player_lives=jnp.array(3),
            current_level=jnp.array(1),
            is_jumping=jnp.array(False),
            step_counter=jnp.array(0),
            jump_counter=jnp.array(0),
            jump_orientation=jnp.array(0),
            is_climbing=jnp.array(False),
            falling_coco_x=jnp.array(0),
            falling_coco_y=jnp.array(0),
            orientation=jnp.array(1),
            jump_base_y=jnp.array(PLAYER_START_Y),
            player_height=jnp.array(PLAYER_HEIGHT),
            climb_base_y=jnp.array(PLAYER_START_Y),
            climb_counter=jnp.array(0),
            punch_left=jnp.array(False),
            punch_right=jnp.array(False)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        (
            player_x,
            player_y,
            vel_x,
            vel_y,
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
        ) = player_step(state, action)

        return State(
            player_x=player_x,
            player_y=player_y,
            player_vel_x=vel_x,
            player_vel_y=vel_y,
            is_crouching=is_crouching,
            player_score=state.player_score,
            player_lives=state.player_lives,
            current_level=state.current_level,
            is_jumping=is_jumping,
            is_climbing=is_climbing,
            step_counter=state.step_counter + 1,
            jump_counter=jump_counter,
            jump_orientation=new_jump_orientation,
            falling_coco_x=state.falling_coco_x,
            falling_coco_y=state.falling_coco_y,
            orientation=orientation,
            jump_base_y=jump_base_y,
            player_height=new_player_height,
            climb_base_y=climb_base_y,
            climb_counter=climb_counter,
            punch_left=punch_left,
            punch_right=punch_right,
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

        # Draw platforms
        if state.current_level == 1:
            for platform in (L1P1, L1P2, L1P3, L1P4):
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

        # Draw Ladders
        if state.current_level == 1:
            for ladder in (L1L1, L1L2, L1L3):
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

        # Draw player
        pygame.draw.rect(
            self.screen,
            PLAYER_COLOR,
            (
                int(state.player_x) * RENDER_SCALE_FACTOR,
                int(state.player_y) * RENDER_SCALE_FACTOR,
                PLAYER_WIDTH * RENDER_SCALE_FACTOR,
                state.player_height * RENDER_SCALE_FACTOR,
            ),
        )

        if state.punch_left:
            pygame.draw.rect(
                self.screen,
                PLAYER_COLOR,
                (
                    int(state.player_x - 2) * RENDER_SCALE_FACTOR,
                    (int(state.player_y) + 8)* RENDER_SCALE_FACTOR,
                    2 * RENDER_SCALE_FACTOR,
                    4 * RENDER_SCALE_FACTOR,
                ),
            )

        if state.punch_right:
            pygame.draw.rect(
                self.screen,
                PLAYER_COLOR,
                (
                    int(state.player_x + PLAYER_WIDTH) * RENDER_SCALE_FACTOR,
                    (int(state.player_y) + 8)* RENDER_SCALE_FACTOR,
                    2 * RENDER_SCALE_FACTOR,
                    4 * RENDER_SCALE_FACTOR,
                ),
            )


        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {state.player_score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        orient_text = font.render(
            f"Orientation: {'left (-1)' if state.orientation < 0 else 'right (1)'}",
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
