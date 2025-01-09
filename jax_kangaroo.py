from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import numpy as np
import pygame

# Test

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


def get_human_action() -> chex.Array:
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_w] or keys[pygame.K_UP]
    down = keys[pygame.K_s] or keys[pygame.K_DOWN]
    left = keys[pygame.K_a] or keys[pygame.K_LEFT]
    right = keys[pygame.K_d] or keys[pygame.K_RIGHT]
    fire = keys[pygame.K_SPACE]

    if up and right and fire:
        return jnp.array(UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(DOWNLEFTFIRE)
    if up and fire:
        return jnp.array(UPFIRE)
    if down and fire:
        return jnp.array(DOWNFIRE)
    if left and fire:
        return jnp.array(LEFTFIRE)
    if right and fire:
        return jnp.array(RIGHTFIRE)
    if up and right:
        return jnp.array(UPRIGHT)
    if up and left:
        return jnp.array(UPLEFT)
    if down and right:
        return jnp.array(DOWNRIGHT)
    if down and left:
        return jnp.array(DOWNLEFT)
    if up:
        return jnp.array(UP)
    if down:
        return jnp.array(DOWN)
    if left:
        return jnp.array(LEFT)
    if right:
        return jnp.array(RIGHT)
    if fire:
        return jnp.array(FIRE)
    return jnp.array(NOOP)


def get_player_platform(state: State) -> Tuple[chex.Array]:
    """
    Returns the platform the player is currently on or has to be on if he is currently jumping i guess o-o
    """
    player_y = state.player_y

    on_p1 = jnp.where(
        jnp.logical_and(
            player_y <= (L1P1.y - PLAYER_HEIGHT), player_y > (L1P2.y - PLAYER_HEIGHT)
        ),
        True,
        False,
    )

    on_p2 = jnp.where(
        jnp.logical_and(
            player_y <= (L1P2.y - PLAYER_HEIGHT), player_y > (L1P3.y - PLAYER_HEIGHT)
        ),
        True,
        False,
    )

    on_p3 = jnp.where(
        jnp.logical_and(
            player_y <= (L1P3.y - PLAYER_HEIGHT), player_y > (L1P4.y - PLAYER_HEIGHT)
        ),
        True,
        False,
    )

    on_p4 = jnp.where(
        player_y <= (L1P4.y - PLAYER_HEIGHT),
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
        on_p1, jnp.where(player_y == (L1P1.y - PLAYER_HEIGHT), True, False), False
    )
    on_p2_ground = jnp.where(
        on_p2, jnp.where(player_y == (L1P2.y - PLAYER_HEIGHT), True, False), False
    )
    on_p3_ground = jnp.where(
        on_p3, jnp.where(player_y == (L1P3.y - PLAYER_HEIGHT), True, False), False
    )
    on_p4_ground = jnp.where(
        on_p4, jnp.where(player_y == (L1P4.y - PLAYER_HEIGHT), True, False), False
    )

    return jnp.any(jnp.array([on_p1_ground, on_p2_ground, on_p3_ground, on_p4_ground]))


def player_jump_controller(
    state: State, jump_pressed: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    player_y = state.player_y
    jump_counter = state.jump_counter
    is_jumping = state.is_jumping

    # Constants
    JUMP_DURATION = 48
    JUMP_STEPS = jnp.array([8, 16, 24, 32, 40])
    JUMP_HEIGHTS = jnp.array([1, 0, -1, -1, -1])  # Scaled jump heights

    # Start jumping if W is pressed and not already jumping
    jump_start = jump_pressed & ~is_jumping
    jump_counter = jnp.where(jump_start, 0, jump_counter)
    is_jumping = is_jumping | jump_start

    # Update jump counter
    jump_counter = jnp.where(is_jumping, jump_counter + 1, jump_counter)

    # Calculate jump offset
    jump_offset = jnp.sum(jnp.where(jump_counter >= JUMP_STEPS, JUMP_HEIGHTS, 0))

    # Apply jump offset to player_y
    player_y = jnp.where(is_jumping, player_y - jump_offset, player_y)

    # Check if jump is complete
    jump_complete = jump_counter >= JUMP_DURATION
    is_jumping = jnp.where(jump_complete, False, is_jumping)
    jump_counter = jnp.where(jump_complete, 0, jump_counter)

    return player_y, jump_counter, is_jumping


def player_step(
    state: State, action: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    x, y = state.player_x, state.player_y
    vel_x, vel_y = state.player_vel_x, state.player_vel_y

    # Handle horizontal movement
    press_left = jnp.any(
        jnp.array([action == LEFT, action == UPLEFT, action == DOWNLEFT])
    )
    press_right = jnp.any(
        jnp.array([action == RIGHT, action == UPRIGHT, action == DOWNRIGHT])
    )
    vel_x = jnp.where(
        press_left, -MOVEMENT_SPEED, jnp.where(press_right, MOVEMENT_SPEED, 0)
    )

    orientation = jnp.where(vel_x < 0, -1, jnp.where(vel_x > 0, 1, state.orientation))

    # Handle vertical movement
    press_up = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT]))

    # on_ground = player_on_ground(state, action)

    y, jump_counter, is_jumping = player_jump_controller(state, press_up)

    # Update vertical velocity based on jumping
    # vel_y = jnp.where(
    #     action == NOOP,
    #     vel_y + GRAVITY,
    #     vel_y,
    # )

    # Update position
    x = jnp.clip(x + vel_x, LEFT_CLIP, RIGHT_CLIP - PLAYER_WIDTH)

    on_p1, on_p2, on_p3, on_p4 = get_player_platform(state)
    y = jax.lax.cond(
        on_p1, lambda y: jnp.clip(y, 0, L1P1.y - PLAYER_HEIGHT), lambda _: _, y
    )
    y = jax.lax.cond(
        on_p2, lambda y: jnp.clip(y, 0, L1P2.y - PLAYER_HEIGHT), lambda _: _, y
    )
    y = jax.lax.cond(
        on_p3, lambda y: jnp.clip(y, 0, L1P3.y - PLAYER_HEIGHT), lambda _: _, y
    )
    y = jax.lax.cond(
        on_p4, lambda y: jnp.clip(y, 0, L1P4.y - PLAYER_HEIGHT), lambda _: _, y
    )

    jax.debug.print("x: {x}, y: {y}", x=x, y=y)

    return x, y, vel_x, vel_y, is_jumping, jump_counter, orientation


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
            is_climbing=jnp.array(False),
            falling_coco_x=jnp.array(0),
            falling_coco_y=jnp.array(0),
            orientation=jnp.array(1),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        player_x, player_y, vel_x, vel_y, is_jumping, jump_counter, orientation = (
            player_step(state, action)
        )
        # enemy_x, enemy_y = enemy_step(state)

        return State(
            player_x=player_x,
            player_y=player_y,
            player_vel_x=vel_x,
            player_vel_y=vel_y,
            player_score=state.player_score,
            player_lives=state.player_lives,
            current_level=state.current_level,
            is_jumping=is_jumping,
            step_counter=state.step_counter + 1,
            jump_counter=jump_counter,
            is_climbing=state.is_climbing,
            falling_coco_x=state.falling_coco_x,
            falling_coco_y=state.falling_coco_y,
            orientation=orientation,
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
                PLAYER_HEIGHT * RENDER_SCALE_FACTOR,
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
