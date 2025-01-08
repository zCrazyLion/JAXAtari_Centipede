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
PLAYER_WIDTH, PLAYER_HEIGHT = 16, 24
ENEMY_WIDTH, ENEMY_HEIGHT = 16, 24
FRUIT_SIZE = 8
APPLE_SIZE = 8

# Colors
BACKGROUND_COLOR = (66, 72, 200)
PLAYER_COLOR = (214, 92, 92)
ENEMY_COLOR = (236, 200, 96)
FRUIT_COLOR = (92, 186, 92)
PLATFORM_COLOR = (236, 236, 236)
APPLE_COLOR = (255, 0, 0)

# Initial positions and physics
PLAYER_START_X, PLAYER_START_Y = 20, 150
GRAVITY = 0.5
JUMP_VELOCITY = -8


class State(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_vel_x: chex.Array
    player_vel_y: chex.Array
    player_score: chex.Array
    player_lives: chex.Array
    fruits: chex.Array
    current_level: chex.Array
    is_jumping: chex.Array
    is_climbing: chex.Array
    step_counter: chex.Array
    falling_coco_x: chex.Array
    falling_coco_y: chex.Array


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


def player_step(
    state: State, action: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    x, y = state.player_x, state.player_y
    vel_x, vel_y = state.player_vel_x, state.player_vel_y

    # Handle horizontal movement
    moving_left = jnp.any(
        jnp.array([action == LEFT, action == UPLEFT, action == DOWNLEFT])
    )
    moving_right = jnp.any(
        jnp.array([action == RIGHT, action == UPRIGHT, action == DOWNRIGHT])
    )
    vel_x = jnp.where(moving_left, -2, jnp.where(moving_right, 2, 0))

    # Handle vertical movement
    up_pressed = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT]))

    on_ground = jnp.equal(y, 186)

    # Determine if the player is jumping
    is_jumping = jnp.logical_or(up_pressed, action == FIRE)

    # Update vertical velocity based on jumping
    vel_y = jnp.where(
        is_jumping,
        jnp.where(on_ground, JUMP_VELOCITY, vel_y + GRAVITY),
        vel_y + GRAVITY,
    )

    # Update position
    x = jnp.clip(x + vel_x, 0, SCREEN_WIDTH - PLAYER_WIDTH)
    y = jnp.clip(y + vel_y, 0, SCREEN_HEIGHT - PLAYER_HEIGHT)

    jax.debug.print("x: {x}, y: {y}", x=x, y=y)

    return x, y, vel_x, vel_y


def collect_fruits(state: State) -> Tuple[chex.Array, chex.Array]:
    player_rect = jnp.array(
        [state.player_x, state.player_y, PLAYER_WIDTH, PLAYER_HEIGHT]
    )
    collected = jnp.any(
        jnp.all(jnp.abs(state.fruits - player_rect[:2]) < FRUIT_SIZE, axis=1)
    )
    score_increase = jnp.where(collected, 100, 0)
    new_fruits = jnp.where(collected, jnp.array([0, 0]), state.fruits)
    return new_fruits, score_increase


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
            fruits=jnp.array([[50, 50], [100, 100], [150, 150], [75, 75], [125, 125]]),
            current_level=jnp.array(1),
            is_jumping=jnp.array(False),
            step_counter=jnp.array(0),
            is_climbing=jnp.array(False),
            falling_coco_x=jnp.array(0),
            falling_coco_y=jnp.array(0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        player_x, player_y, vel_x, vel_y = player_step(state, action)
        # enemy_x, enemy_y = enemy_step(state)

        return State(
            player_x=player_x,
            player_y=player_y,
            player_vel_x=vel_x,
            player_vel_y=vel_y,
            player_score=state.player_score,
            player_lives=state.player_lives,
            fruits=state.fruits,
            current_level=state.current_level,
            is_jumping=jnp.logical_and(vel_y < 0, ~state.is_climbing),
            step_counter=state.step_counter + 1,
            is_climbing=state.is_climbing,
            falling_coco_x=state.falling_coco_x,
            falling_coco_y=state.falling_coco_y,
        )


class Renderer:
    def __init__(self):
        self.screen = pygame.display.set_mode(
            (SCREEN_WIDTH * RENDER_SCALE_FACTOR, SCREEN_HEIGHT * RENDER_SCALE_FACTOR)
        )
        pygame.display.set_caption("Kangaroo")

    def render(self, state: State):
        self.screen.fill(BACKGROUND_COLOR)

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

        # Draw fruits
        for fruit in state.fruits:
            if fruit[0] != 0:
                pygame.draw.rect(
                    self.screen,
                    FRUIT_COLOR,
                    (
                        int(fruit[0]) * RENDER_SCALE_FACTOR,
                        int(fruit[1]) * RENDER_SCALE_FACTOR,
                        FRUIT_SIZE * RENDER_SCALE_FACTOR,
                        FRUIT_SIZE * RENDER_SCALE_FACTOR,
                    ),
                )

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {state.player_score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

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
