from typing import NamedTuple, Tuple

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

# Constants for ball physics
BASE_BALL_SPEED = 1.0
BALL_BOOST_MULTIPLIER = 2.0  # Original Atari doubles speed when fire pressed
BALL_HIT_SPEEDUP = 0.1  # Small speed increase after each hit
BALL_MAX_SPEED = 2.0  # Maximum ball speed cap

# constants for paddle speed influence
MIN_BALL_SPEED = 1.0
PADDLE_SPEED_INFLUENCE = 0.5  # How much paddle speed affects ball velocity
MAX_SPEED_FROM_PADDLE = 1.5  # Maximum additional speed from paddle movement

# Action constants
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3
RIGHTFIRE = 4
LEFTFIRE = 5

BALL_START_X = jnp.array(78)
BALL_START_Y = jnp.array(115)

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
    10: "step_counter",
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


def player_step(state_player_y, state_player_speed, action: chex.Array):
    player_speed = jax.lax.cond(
        jnp.logical_or(action == LEFT, action == LEFTFIRE),
        lambda s: s - PLAYER_ACCELERATION,
        lambda s: jax.lax.cond(
            jnp.logical_or(action == RIGHT, action == RIGHTFIRE),
            lambda s: s + PLAYER_ACCELERATION,
            lambda s: s * 0.9,
            operand=state_player_speed,
        ),
        state_player_speed,
    )

    player_speed = jnp.clip(player_speed, -PLAYER_MAX_SPEED, PLAYER_MAX_SPEED)

    player_y = state_player_y + player_speed

    # check that the player is within the bounds of the game
    player_y = jnp.clip(player_y, WALL_TOP_Y + WALL_TOP_HEIGHT - 8, WALL_BOTTOM_Y - 4)
    player_y = jnp.round(player_y)
    return player_y, player_speed


def ball_step(
    state: State,
    action,
):
    # update the balls position
    ball_x = state.ball_x + state.ball_vel_x
    ball_y = state.ball_y + state.ball_vel_y

    wall_bounce = jnp.logical_or(
        ball_y <= WALL_TOP_Y + WALL_TOP_HEIGHT,
        ball_y >= WALL_BOTTOM_Y - BALL_SIZE[1],
    )
    # calculate bounces on top and bottom walls
    ball_vel_y = jnp.where(wall_bounce, -state.ball_vel_y, state.ball_vel_y)

    # Calculate paddle hits
    player_paddle_hit = jnp.logical_and(
        jnp.logical_and(PLAYER_X <= ball_x, ball_x <= PLAYER_X + PLAYER_SIZE[0]),
        state.ball_vel_x > 0,
    )

    player_paddle_hit = jnp.logical_and(
        player_paddle_hit,
        jnp.logical_and(
            state.player_y - BALL_SIZE[1] <= ball_y,
            ball_y <= state.player_y + PLAYER_SIZE[1] + BALL_SIZE[1],
        ),
    )

    enemy_paddle_hit = jnp.logical_and(
        jnp.logical_and(ENEMY_X <= ball_x, ball_x <= ENEMY_X + ENEMY_SIZE[0]),
        state.ball_vel_x < 0,
    )

    enemy_paddle_hit = jnp.logical_and(
        enemy_paddle_hit,
        jnp.logical_and(
            state.enemy_y - BALL_SIZE[1] <= ball_y,
            ball_y <= state.enemy_y + ENEMY_SIZE[1] + BALL_SIZE[1],
        ),
    )

    paddle_hit = jnp.logical_or(player_paddle_hit, enemy_paddle_hit)

    # Calculate hit position influence (-1 to 1)
    hit_position = jnp.where(
        paddle_hit,
        jnp.where(
            player_paddle_hit,
            (ball_y - state.player_y) / PLAYER_SIZE[1],
            (ball_y - state.enemy_y) / ENEMY_SIZE[1],
        ),
        0.0,
    )

    # Get relevant paddle speed based on which paddle was hit
    paddle_speed = jnp.where(
        player_paddle_hit,
        jnp.abs(state.player_speed),  # Player paddle speed
        jnp.where(
            enemy_paddle_hit,
            jnp.abs(state.enemy_speed),  # Enemy paddle speed
            0.0,  # No hit
        ),
    )

    # Calculate speed addition from paddle movement
    speed_from_paddle = jnp.minimum(
        paddle_speed * PADDLE_SPEED_INFLUENCE, MAX_SPEED_FROM_PADDLE
    )

    # Calculate base speed increase
    current_speed = jnp.abs(state.ball_vel_x)
    base_speed_increase = jnp.where(paddle_hit, BALL_HIT_SPEEDUP, 0.0)

    # Combine all speed factors
    new_speed = jnp.where(
        paddle_hit,
        jnp.minimum(
            current_speed + base_speed_increase + speed_from_paddle, BALL_MAX_SPEED
        ),
        current_speed,
    )

    # Apply boost multiplier if fire button pressed (only for player hits)
    boost_multiplier = jnp.where(
        jnp.logical_and(
            player_paddle_hit,
            jnp.logical_or(
                jnp.logical_or(action == LEFTFIRE, action == RIGHTFIRE),
                action == FIRE,
            ),
        ),
        BALL_BOOST_MULTIPLIER,
        1.0,
    )

    # Calculate final velocities
    ball_vel_y = jnp.where(
        paddle_hit, state.ball_vel_y + (hit_position * 0.5), ball_vel_y
    )
    ball_vel_y = jnp.clip(ball_vel_y, -2.0, 2.0)

    ball_vel_x = jax.lax.cond(
        paddle_hit,
        lambda s: -jnp.sign(s) * new_speed * boost_multiplier,
        lambda s: s.astype(jnp.float32),
        operand=state.ball_vel_x,
    )

    # Ensure minimum ball speed
    ball_vel_x = jnp.where(
        jnp.abs(ball_vel_x) < MIN_BALL_SPEED,
        jnp.sign(ball_vel_x) * MIN_BALL_SPEED,
        ball_vel_x,
    )
    return ball_x, ball_y, ball_vel_x, ball_vel_y


def enemy_step(state_enemy_y, state_enemy_speed, step_counter, ball_y):
    # update the enemy paddle by first checking if this is the 8th step and then updating the speed depending on the ball position
    enemy_speed = jax.lax.cond(
        step_counter % 8,
        lambda s: jax.lax.cond(
            jnp.sign(ball_y - state_enemy_y) < 0,
            lambda x: x - ENEMY_ACCELERATION,
            lambda x: jax.lax.cond(
                jnp.sign(ball_y - state_enemy_y) > 0,
                lambda y: y + ENEMY_ACCELERATION,
                lambda y: y * 0.9,
                operand=x,
            ),
            operand=s,
        ),
        lambda s: s,
        operand=state_enemy_speed,
    )

    # limit the enemy speed to the maximum allowed value
    enemy_speed = jnp.clip(enemy_speed, -ENEMY_MAX_SPEED, ENEMY_MAX_SPEED)

    # update the enemy position
    enemy_y = state_enemy_y + enemy_speed

    # check collision with the walls
    enemy_y = jnp.clip(enemy_y, WALL_TOP_Y + WALL_TOP_HEIGHT - 8, WALL_BOTTOM_Y - 4)
    return enemy_y, enemy_speed


def _reset_ball_after_goal(
    state_and_goal: Tuple[State, bool]
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """
    Determines new ball position and velocity after a goal.
    Args:
        state_and_goal: Tuple of (current state, whether goal was scored on right side)
    Returns:
        Tuple of (ball_x, ball_y, ball_vel_x, ball_vel_y) as int32 arrays
    """
    state, scored_right = state_and_goal

    # Determine Y velocity direction based on ball position
    ball_vel_y = jnp.where(
        state.ball_y > BALL_START_Y,
        1,  # Ball was in lower half, go down
        -1,  # Ball was in upper half, go up
    ).astype(jnp.int32)

    # X velocity is always towards the side that just got scored on
    ball_vel_x = jnp.where(
        scored_right, 1, -1  # Ball moves right  # Ball moves left
    ).astype(jnp.int32)

    return (
        BALL_START_X.astype(jnp.float32),
        BALL_START_Y.astype(jnp.float32),
        ball_vel_x.astype(jnp.float32),
        ball_vel_y.astype(jnp.float32),
    )


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
            step_counter=jnp.array(0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        player_speed = jax.lax.cond(
            jnp.logical_or(action == LEFT, action == LEFTFIRE),
            lambda s: s - PLAYER_ACCELERATION,
            lambda s: jax.lax.cond(
                jnp.logical_or(action == RIGHT, action == RIGHTFIRE),
                lambda s: s + PLAYER_ACCELERATION,
                lambda s: s * 0.9,
                operand=state.player_speed,
            ),
            state.player_speed,
        )

        player_speed = jnp.clip(player_speed, -PLAYER_MAX_SPEED, PLAYER_MAX_SPEED)

        player_y = state.player_y + player_speed

        # check that the player is within the bounds of the game
        player_y = jnp.clip(
            player_y, WALL_TOP_Y + WALL_TOP_HEIGHT - 8, WALL_BOTTOM_Y - 4
        )
        player_y = jnp.round(player_y)

        # update the balls position
        ball_x = state.ball_x + state.ball_vel_x
        ball_y = state.ball_y + state.ball_vel_y

        wall_bounce = jnp.logical_or(
            ball_y <= WALL_TOP_Y + WALL_TOP_HEIGHT,
            ball_y >= WALL_BOTTOM_Y - BALL_SIZE[1],
        )
        # calculate bounces on top and bottom walls
        ball_vel_y = jnp.where(wall_bounce, -state.ball_vel_y, state.ball_vel_y)

        paddle_bounce = jnp.logical_and(
            jnp.logical_and(PLAYER_X <= ball_x, ball_x <= PLAYER_X + PLAYER_SIZE[0]),
            state.ball_vel_x > 0,
        )

        # also check if the y position is within the player paddle
        paddle_bounce = jnp.logical_and(
            paddle_bounce,
            jnp.logical_and(
                state.player_y - BALL_SIZE[1] <= ball_y,
                ball_y <= state.player_y + PLAYER_SIZE[1] + BALL_SIZE[1],
            ),
        )

        # Apply speed boost if spacebar is pressed during paddle hit
        boost_multiplier = jnp.where(
            jnp.logical_and(
                paddle_bounce, jnp.logical_or(action == FIRE, action == RIGHTFIRE)
            ),
            BALL_BOOST_MULTIPLIER,
            1.0,
        )

        # calculate bounces on player paddle
        ball_vel_x = jax.lax.cond(
            paddle_bounce,
            lambda s: jnp.array(-s * boost_multiplier, float),
            lambda s: jnp.array(s, float),
            operand=state.ball_vel_x,
        )

        ball_vel_x = jnp.where(
            jnp.logical_and(
                paddle_bounce,
                jnp.logical_or(
                    jnp.logical_or(action == LEFTFIRE, action == RIGHTFIRE),
                    action == FIRE,
                ),
            ),
            ball_vel_x * 2,
            ball_vel_x,
        )

        paddle_bounce = jnp.logical_and(
            jnp.logical_and(ENEMY_X <= ball_x, ball_x <= ENEMY_X + ENEMY_SIZE[0]),
            state.ball_vel_x < 0,
        )

        # also check if the y position is within the enemy paddle
        paddle_bounce = jnp.logical_and(
            paddle_bounce,
            jnp.logical_and(
                state.enemy_y - BALL_SIZE[1] <= ball_y,
                ball_y <= state.enemy_y + ENEMY_SIZE[1] + BALL_SIZE[1],
            ),
        )

        # calculate bounces on enemy paddle
        ball_vel_x = jax.lax.cond(
            paddle_bounce, lambda s: -s, lambda s: s, operand=ball_vel_x
        )

        # Score and goal detection
        player_goal = ball_x < ENEMY_X - ENEMY_SIZE[0]
        enemy_goal = ball_x > PLAYER_X + PLAYER_SIZE[0]
        ball_reset = jnp.logical_or(enemy_goal, player_goal)

        # Update scores
        player_score = jax.lax.cond(
            player_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.player_score,
        )
        enemy_score = jax.lax.cond(
            enemy_goal,
            lambda s: s + 1,
            lambda s: s,
            operand=state.enemy_score,
        )

        # Get final ball values accounting for reset
        current_values = (
            ball_x.astype(jnp.float32),
            ball_y.astype(jnp.float32),
            ball_vel_x.astype(jnp.float32),
            ball_vel_y.astype(jnp.float32),
        )
        ball_x_final, ball_y_final, ball_vel_x_final, ball_vel_y_final = jax.lax.cond(
            ball_reset,
            lambda x: _reset_ball_after_goal((state, enemy_goal)),
            lambda x: x,
            operand=current_values,
        )

        # Enemy paddle AI movement
        enemy_speed = jax.lax.cond(
            state.step_counter % 8,
            lambda s: jax.lax.cond(
                jnp.sign(ball_y_final - state.enemy_y) < 0,
                lambda x: x - ENEMY_ACCELERATION,
                lambda x: jax.lax.cond(
                    jnp.sign(ball_y_final - state.enemy_y) > 0,
                    lambda y: y + ENEMY_ACCELERATION,
                    lambda y: y * 0.9,
                    operand=x,
                ),
                operand=s,
            ),
            lambda s: s,
            operand=state.enemy_speed,
        )

        enemy_speed = jnp.clip(enemy_speed, -ENEMY_MAX_SPEED, ENEMY_MAX_SPEED)
        enemy_y = jnp.clip(
            state.enemy_y + enemy_speed,
            WALL_TOP_Y + WALL_TOP_HEIGHT - 8,
            WALL_BOTTOM_Y - 4,
        )

        # Reset enemy position on goal
        enemy_y_final = jax.lax.cond(
            ball_reset,
            lambda s: BALL_START_Y.astype(jnp.float32),
            lambda s: enemy_y.astype(jnp.float32),
            operand=None,
        )

        return State(
            player_y=player_y,
            player_speed=player_speed,
            ball_x=ball_x_final,  # Use final values that include reset
            ball_y=ball_y_final,  # Use final values that include reset
            enemy_y=enemy_y_final,  # Use final enemy position
            enemy_speed=enemy_speed,
            ball_vel_x=ball_vel_x_final,  # Use final values that include reset
            ball_vel_y=ball_vel_y_final,  # Use final values that include reset
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=state.step_counter + 1,
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
            np.rot90(canvas_np, k=1)
        )  # Rotate 90 degrees counterclockwise and flip vertically to fix orientation
        pygame_surface = pygame.surfarray.make_surface(canvas_np)
        screen.blit(
            pygame.transform.scale(pygame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)),
            (0, 0),
        )
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
            canvas[
                int(state["player_y"]) : int(state["player_y"]) + PLAYER_SIZE[1],
                PLAYER_X : PLAYER_X + PLAYER_SIZE[0],
            ] = PLAYER_COLOR  # Player paddle
        if 0 <= int(state["enemy_y"]) < canvas.shape[0] - ENEMY_SIZE[1]:
            canvas[
                int(state["enemy_y"]) : int(state["enemy_y"]) + ENEMY_SIZE[1],
                ENEMY_X : ENEMY_X + ENEMY_SIZE[0],
            ] = ENEMY_COLOR  # Enemy paddle
        if (
            0 <= int(state["ball_y"]) < canvas.shape[0] - BALL_SIZE[1]
            and 0 <= int(int(state["ball_x"])) < canvas.shape[1] - BALL_SIZE[0]
        ):
            canvas[
                int(state["ball_y"]) : int(state["ball_y"]) + BALL_SIZE[1],
                int(state["ball_x"]) : int(state["ball_x"]) + BALL_SIZE[0],
            ] = BALL_COLOR  # Ball

        # Draw walls
        canvas[WALL_TOP_Y : WALL_TOP_Y + WALL_TOP_HEIGHT, :] = WALL_COLOR  # Top wall
        canvas[WALL_BOTTOM_Y : WALL_BOTTOM_Y + WALL_BOTTOM_HEIGHT, :] = (
            WALL_COLOR  # Bottom wall
        )

        # Draw scores
        self.draw_score(
            canvas, state["player_score"], position=(120, 2), color=PLAYER_COLOR
        )
        self.draw_score(
            canvas, state["enemy_score"], position=(30, 2), color=ENEMY_COLOR
        )

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
                            for dj in range(
                                4
                            ):  # Zoom each pixel by 4 times horizontally
                                canvas[y_offset + i * 4 + di, x_offset + j * 4 + dj] = (
                                    color
                                )
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
