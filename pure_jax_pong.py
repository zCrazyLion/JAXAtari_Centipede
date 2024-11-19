from typing import Tuple, NamedTuple

import jax.lax
import jax.numpy as jnp
import chex
import numpy as np
import pygame
from numbers_impl import digits
from functools import partial

# Constants for game environment
PLAYER_ACCELERATION = 0.2
PLAYER_MAX_SPEED = 2.0
BALL_SPEED = jnp.array([1, 1])  # Ball speed in x and y direction
ENEMY_ACCELERATION = 0.2
ENEMY_MAX_SPEED = 2.0

# Action constants
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
RIGHTFIRE = 4
LEFTFIRE = 5

# Background color and object colors
BACKGROUND_COLOR = 144, 72, 17
PLAYER_COLOR = 92, 186, 92
ENEMY_COLOR = 213, 130, 74
BALL_COLOR = 236, 236, 236  # White ball
WALL_COLOR = 236, 236, 236  # White walls
SCORE_COLOR = 236, 236, 236  # White score

# Player and enemy paddle positions
PLAYER_X = 140
ENEMY_X = 16

# Object sizes (width, height)
PLAYER_SIZE = (4, 15)
BALL_SIZE = (2, 4)
ENEMY_SIZE = (4, 15)
WALL_TOP_Y = 24
WALL_TOP_HEIGHT = 9
WALL_BOTTOM_Y = 194
WALL_BOTTOM_HEIGHT = 16

# Pygame window dimensions
WINDOW_WIDTH = 160 * 4
WINDOW_HEIGHT = 210 * 4

# define the positions of the state information
# define the positions of the state information
STATE_TRANSLATOR: dict = {
    0: "player_y",
    1: "player_speed",
    2: "ball_x",
    3: "ball_y",
    4: "enemy_y",
    5: "enemy_speed",
    6: "ball_vel_x",
    7: "ball_vel_y",
    8: "player_score",
    9: "enemy_score",
    10: "step_counter"
}


def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return jnp.array(LEFTFIRE)
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return jnp.array(RIGHTFIRE)
    elif keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    else:
        return jnp.array(NOOP)


# immutable state container
class State(NamedTuple):
    player_y: chex.Array
    player_speed: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    enemy_y: chex.Array
    enemy_speed: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    step_counter: chex.Array


# TODO: how far do the competencies of the environment extend? Does it need to handle rewards and timesteps?
class Game:
    def __init__(self):
        pass

    def reset(self) -> State:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        return State(
            player_y=jnp.array(96),
            player_speed=jnp.array(0.0),
            ball_x=jnp.array(78),
            ball_y=jnp.array(115),
            enemy_y=jnp.array(115),
            enemy_speed=jnp.array(0.0),
            ball_vel_x=BALL_SPEED[0],
            ball_vel_y=BALL_SPEED[1],
            player_score=jnp.array(0),
            enemy_score=jnp.array(0),
            step_counter=jnp.array(0)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        player_speed = jax.lax.cond(
            action == LEFT,
            lambda s: s - PLAYER_ACCELERATION,
            lambda s: jax.lax.cond(
                action == RIGHT,
                lambda s: s + PLAYER_ACCELERATION,
                lambda s: s * 0.9,
                operand=state.player_speed
            ), state.player_speed)

        player_speed = jnp.clip(player_speed, -PLAYER_MAX_SPEED, PLAYER_MAX_SPEED)

        player_y = state.player_y + player_speed

        # check that the player is within the bounds of the game
        player_y = jnp.clip(player_y, WALL_TOP_Y + WALL_TOP_HEIGHT - 8, WALL_BOTTOM_Y - 4)
        player_y = jnp.round(player_y)

        # update the balls position
        ball_x = state.ball_x + state.ball_vel_x
        ball_y = state.ball_y + state.ball_vel_y

        wall_bounce = jnp.logical_or(
            ball_y <= WALL_TOP_Y + WALL_TOP_HEIGHT,
            ball_y >= WALL_BOTTOM_Y - BALL_SIZE[1]
        )
        # calculate bounces on top and bottom walls
        ball_vel_y = jnp.where(wall_bounce, -state.ball_vel_y, state.ball_vel_y)

        paddle_bounce = jnp.logical_and(
            jnp.logical_and(
                PLAYER_X <= ball_x,
                ball_x <= PLAYER_X + PLAYER_SIZE[0]
            ),
            state.ball_vel_x > 0
        )

        # also check if the y position is within the player paddle
        paddle_bounce = jnp.logical_and(
            paddle_bounce,
            jnp.logical_and(
                state.player_y - BALL_SIZE[1] <= ball_y,
                ball_y <= state.player_y + PLAYER_SIZE[1] + BALL_SIZE[1]
            )
        )

        # calculate bounces on player paddle
        ball_vel_x = jax.lax.cond(
            paddle_bounce,
            lambda s: -s,
            lambda s: s,
            operand=state.ball_vel_x
        )

        paddle_bounce = jnp.logical_and(
            jnp.logical_and(
                ENEMY_X <= ball_x,
                ball_x <= ENEMY_X + ENEMY_SIZE[0]
            ),
            state.ball_vel_x < 0
        )

        # also check if the y position is within the enemy paddle
        paddle_bounce = jnp.logical_and(
            paddle_bounce,
            jnp.logical_and(
                state.enemy_y - BALL_SIZE[1] <= ball_y,
                ball_y <= state.enemy_y + ENEMY_SIZE[1] + BALL_SIZE[1]
            )
        )

        # calculate bounces on enemy paddle
        ball_vel_x = jax.lax.cond(
            paddle_bounce,
            lambda s: -s,
            lambda s: s,
            operand=ball_vel_x
        )

        # calculate score changes
        enemy_score = jax.lax.cond(
            ball_x > PLAYER_X + PLAYER_SIZE[0],
            lambda s: s + 1,
            lambda s: s,
            operand=state.enemy_score
        )

        player_score = jax.lax.cond(
            ball_x < ENEMY_X - ENEMY_SIZE[0],
            lambda s: s + 1,
            lambda s: s,
            operand=state.player_score
        )

        ball_reset = jnp.logical_or(
            enemy_score != state.enemy_score,
            player_score != state.player_score
        )

        # Create a JAX array with the reset values
        reset_values = jnp.array([78, 115, BALL_SPEED[0], BALL_SPEED[1]])
        current_values = jnp.array([ball_x, ball_y, ball_vel_x, ball_vel_y])

        # Use where with the arrays
        new_values = jnp.where(ball_reset, reset_values, current_values)

        # Unpack the values
        ball_x, ball_y, ball_vel_x, ball_vel_y = new_values

        # update the enemy paddle by first checking if this is the 8th step and then updating the speed depending on the ball position
        enemy_speed = jax.lax.cond(
            state.step_counter % 8,
            lambda s: jax.lax.cond(
                jnp.sign(state.ball_y - state.enemy_y) < 0,
                lambda x: x - ENEMY_ACCELERATION,
                lambda x: jax.lax.cond(
                    jnp.sign(state.ball_y - state.enemy_y) > 0,
                    lambda y: y + ENEMY_ACCELERATION,
                    lambda y: y * 0.9,
                    operand=x
                ), operand=s
            ),
            lambda s: s,
            operand=state.enemy_speed
        )

        # limit the enemy speed to the maximum allowed value
        enemy_speed = jnp.clip(enemy_speed, -ENEMY_MAX_SPEED, ENEMY_MAX_SPEED)

        # update the enemy position
        enemy_y = state.enemy_y + enemy_speed

        # check collision with the walls
        enemy_y = jnp.clip(enemy_y, WALL_TOP_Y + WALL_TOP_HEIGHT - 8, WALL_BOTTOM_Y - 4)

        return State(
            player_y=player_y,
            player_speed=player_speed,
            ball_x=ball_x,
            ball_y=ball_y,
            enemy_y=enemy_y,
            enemy_speed=enemy_speed,
            ball_vel_x=ball_vel_x,
            ball_vel_y=ball_vel_y,
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=state.step_counter + 1
        )


class Renderer:
    def __init__(self, jax_translator):
        self.translator = jax_translator

    def display(self, screen, state):
        """
        Displays the rendered game state using pygame.
        """
        canvas = self.jax_rendering(self.convert_jax_arr(state))
        canvas_np = np.array(canvas)
        canvas_np = np.flipud(
            np.rot90(canvas_np, k=1))  # Rotate 90 degrees counterclockwise and flip vertically to fix orientation
        pygame_surface = pygame.surfarray.make_surface(canvas_np)
        screen.blit(pygame.transform.scale(pygame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))
        pygame.display.flip()

    def convert_jax_arr(self, state: jnp.array) -> dict:
        """
        Converts a JAX array to a dict using the translator
        """
        state_dict = {}
        for i in range(len(state)):
            state_dict[self.translator[i]] = state[i]

        return state_dict

    def jax_rendering(self, state: dict):
        """
        Renders the current state of the game
        """
        # Create a blank canvas with background color
        canvas = np.full((210, 160, 3), BACKGROUND_COLOR, dtype=np.uint8)

        # Draw player, ball, and enemy on the canvas
        if 0 <= int(state["player_y"]) < canvas.shape[0] - PLAYER_SIZE[1]:
            canvas[int(state["player_y"]):int(state["player_y"]) + PLAYER_SIZE[1],
            PLAYER_X:PLAYER_X + PLAYER_SIZE[0]] = PLAYER_COLOR  # Player paddle
        if 0 <= int(state["enemy_y"]) < canvas.shape[0] - ENEMY_SIZE[1]:
            canvas[int(state["enemy_y"]):int(state["enemy_y"]) + ENEMY_SIZE[1],
            ENEMY_X:ENEMY_X + ENEMY_SIZE[0]] = ENEMY_COLOR  # Enemy paddle
        if 0 <= int(state["ball_y"]) < canvas.shape[0] - BALL_SIZE[1] and 0 <= int(int(state["ball_x"])) < canvas.shape[
            1] - BALL_SIZE[0]:
            canvas[int(state["ball_y"]):int(state["ball_y"]) + BALL_SIZE[1],
            int(state["ball_x"]):int(state["ball_x"]) + BALL_SIZE[0]] = BALL_COLOR  # Ball

        # Draw walls
        canvas[WALL_TOP_Y:WALL_TOP_Y + WALL_TOP_HEIGHT, :] = WALL_COLOR  # Top wall
        canvas[WALL_BOTTOM_Y:WALL_BOTTOM_Y + WALL_BOTTOM_HEIGHT, :] = WALL_COLOR  # Bottom wall

        # Draw scores
        self.draw_score(canvas, state["player_score"], position=(120, 2), color=PLAYER_COLOR)
        self.draw_score(canvas, state["enemy_score"], position=(30, 2), color=ENEMY_COLOR)

        return jnp.array(canvas)

    def draw_score(self, canvas, score, position, color):
        """
        Draws the score on the canvas.

        Args:
            canvas: The canvas to draw on.
            score: The score to draw.
            position: The (x, y) position to start drawing the score.
            color: The color to use for the score.
        """
        x_offset, y_offset = position
        score_str = str(score)
        for digit_char in score_str:
            digit = digits[int(digit_char)]
            for i in range(digit.shape[0]):
                for j in range(digit.shape[1]):
                    if digit[i, j] == 1:
                        for di in range(4):  # Zoom each pixel by 4 times vertically
                            for dj in range(4):  # Zoom each pixel by 4 times horizontally
                                canvas[y_offset + i * 4 + di, x_offset + j * 4 + dj] = color
            x_offset += 16  # Space between digits (4 times the original space)


if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Pong Game")
    clock = pygame.time.Clock()

    # Create a Game instance
    game = Game()

    # create a Renderer instance
    renderer = Renderer(STATE_TRANSLATOR)

    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_state = jitted_reset()
    # Run the game until the user quits
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = get_human_action()
        curr_state = jitted_step(curr_state, action)
        renderer.display(screen, curr_state)
        clock.tick(60)  # Set frame rate to 60 FPS

    # Quit Pygame
    pygame.quit()
