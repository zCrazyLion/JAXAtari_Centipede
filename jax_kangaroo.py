from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import numpy as np
import pygame

# Action constants
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

# Game constants
RENDER_SCALE_FACTOR = 3
SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210
PLAYER_WIDTH, PLAYER_HEIGHT = 8, 24
ENEMY_WIDTH, ENEMY_HEIGHT = 8, 24
FRUIT_SIZE = 8
APPLE_SIZE = 8

# Colors
BACKGROUND_COLOR = (66, 72, 200)
PLAYER_COLOR = (255, 145, 0)
ENEMY_COLOR = (236, 200, 96)
FRUIT_COLOR = (92, 186, 92)
PLATFORM_COLOR = (130, 74, 0)
APPLE_COLOR = (255, 0, 0)

# Initial positions and physics
PLAYER_START_X, PLAYER_START_Y = 23, 148
GRAVITY = 0.5
JUMP_VELOCITY = -8
MOVEMENT_SPEED = 1

LEFT_CLIP = 16
RIGHT_CLIP = 144


class Platform(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array


# Platform positions
P_HEIGHT = 4

L1P1 = Platform(x=16, y=172, w=128, h=P_HEIGHT)
L1P2 = Platform(x=16, y=124, w=128, h=P_HEIGHT)
L1P3 = Platform(x=16, y=76, w=128, h=P_HEIGHT)
L1P4 = Platform(x=16, y=28, w=128, h=P_HEIGHT)


class State(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_vel_x: chex.Array
    player_vel_y: chex.Array
    player_score: chex.Array
    player_lives: chex.Array
    current_level: chex.Array
    is_jumping: chex.Array
    is_climbing: chex.Array
    step_counter: chex.Array
    jump_counter: chex.Array
    falling_coco_x: chex.Array
    falling_coco_y: chex.Array
    orientation: chex.Array
    jump_base_y: chex.Array
    jump_orientation: chex.Array
    player_height: chex.Array


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

    # def on_platform(i, val):

    #     return jnp.logical_and(
    #         jnp.logical_and(player_x >= platform.x, player_x < (platform.x + platform.w)),
    #         jnp.logical_and(player_y >= platform.y, player_y < (platform.y + platform.h)),
    #     )

    # jax.lax.fori_loop()

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


@jax.jit
def player_jump_controller(
    state: State, jump_pressed: chex.Array
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

    # Start jump if pressed & not already jumping
    jump_start = jump_pressed & ~is_jumping

    jump_counter = jnp.where(jump_start, 0, jump_counter)
    jump_orientation = jnp.where(jump_start, state.orientation, state.jump_orientation)
    jump_base_y = jnp.where(jump_start, player_y, jump_base_y)
    is_jumping = is_jumping | jump_start

    # If currently jumping, increment counter
    jump_counter = jnp.where(is_jumping, jump_counter + 1, jump_counter)

    def offset_for(count):
        # piecewise intervals:
        #   [0..8)    ->  0
        #   [8..16)   -> -12
        #   [16..24)  -> -12
        #   [24..32)  -> -24
        #   [32..40)  -> -12
        #   >= 40     ->  0  (jump ends)
        conditions = [
            (count <= 8),
            (count < 16),
            (count <= 24),
            (count <= 32),
            (count < 41),
        ]
        values = [
            0,  # 0..7
            -8,  # 8..15
            -8,  # 16..23
            -16,  # 24..31
            -8,  # 32..39
        ]
        return jnp.select(conditions, values, default=0)

    total_offset = offset_for(jump_counter)

    # Apply offset only if is_jumping
    new_y = jnp.where(is_jumping, jump_base_y + total_offset, player_y)

    # After 40 ticks, jump is done
    jump_complete = jump_counter >= 41
    is_jumping = jnp.where(jump_complete, False, is_jumping)
    jump_counter = jnp.where(jump_complete, 0, jump_counter)

    return new_y, jump_counter, is_jumping, jump_base_y, jump_orientation


@jax.jit
def player_height_controller(
    is_jumping: chex.Array,
    jump_counter: chex.Array,
    pressing_down: chex.Array,
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
            (count < 16),  # covers 0..15
            (count < 24),  # covers 16..23
            (count < 40),  # covers 24..39
        ]
        values = [
            24,  # normal
            15,  # small sprite
            23,  # stretched
        ]
        return jnp.select(conditions, values, default=24)

    candidate_height = jump_based_height(jump_counter)
    height_if_jumping = jnp.where(is_jumping, candidate_height, 24)

    # Only allow crouching if pressing_down == True AND not jumping
    is_crouching = jnp.logical_and(pressing_down, jnp.logical_not(is_jumping))

    # If can_crouch, override => 16; otherwise use jump-based or normal height
    new_height = jnp.where(is_crouching, 16, height_if_jumping)
    return new_height


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

    # Get inputs
    press_right = jnp.any(
        jnp.array([action == RIGHT, action == UPRIGHT, action == DOWNRIGHT])
    )

    press_left = jnp.any(
        jnp.array([action == LEFT, action == UPLEFT, action == DOWNLEFT])
    )

    press_up = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT]))

    press_down = jnp.any(
        jnp.array([action == DOWN, action == DOWNLEFT, action == DOWNRIGHT])
    )
    press_down = jnp.where(state.is_jumping, False, press_down)

    # Calculate horizontal velocity
    is_crouching = jnp.logical_and(press_down, ~state.is_jumping)
    candidate_vel_x = jnp.where(
        is_crouching,
        0,
        jnp.where(
            press_left, -MOVEMENT_SPEED, jnp.where(press_right, MOVEMENT_SPEED, 0)
        ),
    )

    # Jump controller
    new_y, new_jump_counter, new_is_jumping, new_jump_base_y, new_jump_orientation = (
        player_jump_controller(state, press_up)
    )

    new_orientation = jnp.sign(candidate_vel_x)
    stop_in_air = jnp.logical_and(
        new_is_jumping, state.jump_orientation != new_orientation
    )
    vel_x = jnp.where(stop_in_air, 0, candidate_vel_x)

    # Height controller
    new_player_height = player_height_controller(
        is_jumping=new_is_jumping,
        jump_counter=new_jump_counter,
        pressing_down=press_down,
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
        new_is_jumping,
        new_jump_counter,
        new_orientation,
        new_jump_base_y,
        new_player_height,
        new_jump_orientation,
    )


class Game:
    def __init__(self, frameskip: int = 1):
        self.frameskip = frameskip

    def reset(self) -> State:
        return State(
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_vel_x=jnp.array(0),
            player_vel_y=jnp.array(0),
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
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        (
            player_x,
            player_y,
            vel_x,
            vel_y,
            is_jumping,
            jump_counter,
            orientation,
            jump_base_y,
            new_player_height,
            new_jump_orientation,
        ) = player_step(state, action)

        return State(
            player_x=player_x,
            player_y=player_y,
            player_vel_x=vel_x,
            player_vel_y=vel_y,
            player_score=state.player_score,
            player_lives=state.player_lives,
            current_level=state.current_level,
            is_jumping=is_jumping,
            is_climbing=state.is_climbing,
            step_counter=state.step_counter + 1,
            jump_counter=jump_counter,
            jump_orientation=new_jump_orientation,
            falling_coco_x=state.falling_coco_x,
            falling_coco_y=state.falling_coco_y,
            orientation=orientation,
            jump_base_y=jump_base_y,
            player_height=new_player_height,
        )


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
