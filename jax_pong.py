from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import numpy as np
import pygame

from numbers_impl import digits

# Constants for game environment
MAX_SPEED = 12
BALL_SPEED = jnp.array([-1, 1])  # Ball speed in x and y direction
ENEMY_STEP_SIZE = 2

# Constants for ball physics
BASE_BALL_SPEED = 1
BALL_MAX_SPEED = 4  # Maximum ball speed cap

# constants for paddle speed influence
MIN_BALL_SPEED = 1

PLAYER_ACCELERATION = jnp.array([6, 3, 1, -1, 1, -1, 0, 0, 1, 0, -1, 0, 1])

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
PLAYER_SIZE = (4, 16)
BALL_SIZE = (2, 4)
ENEMY_SIZE = (4, 16)
WALL_TOP_Y = 24
WALL_TOP_HEIGHT = 10
WALL_BOTTOM_Y = 194
WALL_BOTTOM_HEIGHT = 16

# Pygame window dimensions
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

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
    11: "acceleration_counter",
    12: "buffer",
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
    acceleration_counter: chex.Array
    buffer: chex.Array

class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray

class PongObservation(NamedTuple):
    player: EntityPosition
    enemy: EntityPosition
    ball: EntityPosition
    score_player: jnp.ndarray
    score_enemy: jnp.ndarray

class PongInfo(NamedTuple):
    time: jnp.ndarray


def player_step(
    state_player_y, state_player_speed, acceleration_counter, action: chex.Array
):
    # check if one of the buttons is pressed
    up = jnp.logical_or(action == LEFT, action == LEFTFIRE)
    down = jnp.logical_or(action == RIGHT, action == RIGHTFIRE)

    # get the current acceleration
    acceleration = PLAYER_ACCELERATION[acceleration_counter]

    # perform the deceleration checks first, since in the base game
    # on a direction switch the player is first decelerated and then accelerated in the new direction
    # check if the player touches a wall
    touches_wall = jnp.logical_or(
        state_player_y < WALL_TOP_Y,
        state_player_y + PLAYER_SIZE[1] > WALL_BOTTOM_Y,
    )

    player_speed = state_player_speed

    # if no button was clicked OR the paddle touched a wall and there is a speed, apply deceleration (halfing the speed every tick)
    player_speed = jax.lax.cond(
        jnp.logical_or(jnp.logical_not(jnp.logical_or(up, down)), touches_wall),
        lambda s: jnp.round(s / 2).astype(jnp.int32),
        lambda s: s,
        operand=player_speed,
    )

    direction_change_up = jnp.logical_and(up, state_player_speed > 0)
    # also apply deceleration if the direction is changed
    player_speed = jax.lax.cond(
        direction_change_up,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )
    direction_change_down = jnp.logical_and(down, state_player_speed < 0)

    player_speed = jax.lax.cond(
        direction_change_down,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )

    # reset the acceleration counter on a direction change
    direction_change = jnp.logical_or(direction_change_up, direction_change_down)
    acceleration_counter = jax.lax.cond(
        direction_change,
        lambda _: 0,
        lambda s: s,
        operand=acceleration_counter,
    )

    # add the current acceleration to the speed (positive if up, negative if down)
    player_speed = jax.lax.cond(
        up,
        lambda s: jnp.maximum(s - acceleration, -MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    player_speed = jax.lax.cond(
        down,
        lambda s: jnp.minimum(s + acceleration, MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    # reset or increment the acceleration counter here
    new_acceleration_counter = jax.lax.cond(
        jnp.logical_or(up, down),  # If moving in either direction
        lambda s: jnp.minimum(s + 1, 15),  # Increment counter
        lambda s: 0,  # Reset if no movement
        operand=acceleration_counter,
    )

    # calculate the new player position
    player_y = jnp.clip(
        state_player_y + player_speed,
        WALL_TOP_Y + WALL_TOP_HEIGHT - 10,
        WALL_BOTTOM_Y - 4,
    )
    return player_y, player_speed, new_acceleration_counter


def ball_step(
    state: State,
    action,
):
    # update the balls position
    ball_x = state.ball_x + state.ball_vel_x
    ball_y = state.ball_y + state.ball_vel_y

    wall_bounce = jnp.logical_or(
        ball_y <= WALL_TOP_Y + WALL_TOP_HEIGHT - BALL_SIZE[1],
        ball_y >= WALL_BOTTOM_Y,
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
        jnp.logical_and(ENEMY_X <= ball_x, ball_x <= ENEMY_X + ENEMY_SIZE[0] - 1),
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

    # Calculate hit position on paddle (divide paddle into 5 equal sections)
    section_height = PLAYER_SIZE[1] / 5  # Each section is 1/5 of paddle height

    # Calculate relative hit position (int between -2 and 2, which is also the relevant y speed depending on the hit paddle)
    hit_position = jnp.where(
        paddle_hit,
        jnp.where(
            player_paddle_hit,
            # For player paddle
            jnp.where(
                ball_y < state.player_y + section_height,
                -2.0,  # Top section -> strong up
                jnp.where(
                    ball_y < state.player_y + 2 * section_height,
                    -1.0,  # Upper middle -> medium up
                    jnp.where(
                        ball_y < state.player_y + 3 * section_height,
                        0.0,  # Center section -> straight
                        jnp.where(
                            ball_y < state.player_y + 4 * section_height,
                            1.0,  # Lower middle -> medium down
                            2.0,  # Bottom section -> strong down
                        ),
                    ),
                ),
            ),
            # For enemy paddle (same logic)
            jnp.where(
                ball_y < state.enemy_y + section_height,
                -2.0,
                jnp.where(
                    ball_y < state.enemy_y + 2 * section_height,
                    -1.0,
                    jnp.where(
                        ball_y < state.enemy_y + 3 * section_height,
                        0.0,
                        jnp.where(
                            ball_y < state.enemy_y + 4 * section_height,
                            1.0,
                            2.0,
                        ),
                    ),
                ),
            ),
        ),
        0.0,
    )

    # Get relevant paddle speed based on which paddle was hit
    paddle_speed = jnp.where(
        player_paddle_hit,
        state.player_speed,
        jnp.where(
            enemy_paddle_hit,
            state.enemy_speed,
            0.0,
        ),
    )

    # Calculate new y velocity
    ball_vel_y = jnp.where(paddle_hit, hit_position, ball_vel_y)

    # calculate the new ball_vel_x position depending on 1. if a boost was hit or 2. the ball was hit with max velocity by the player (eval tbd?)
    # first check the paddle
    boost_triggered = jnp.logical_and(
        player_paddle_hit,
        jnp.logical_or(
            jnp.logical_or(action == LEFTFIRE, action == RIGHTFIRE),
            action == FIRE,
        ),
    )
    # and check if the paddle hit the ball at MAX speed
    player_max_hit = jnp.logical_and(player_paddle_hit, state.player_speed == MAX_SPEED)
    # if any of the two is true, increase/decrease the ball_vel_x by 1 based on current direction
    ball_vel_x = jnp.where(
        jnp.logical_or(boost_triggered, player_max_hit),
        state.ball_vel_x
        + jnp.sign(state.ball_vel_x),  # Add/subtract 1 based on direction
        state.ball_vel_x,
    )

    # invert ball_vel_x if a paddle was hit
    ball_vel_x = jnp.where(
        paddle_hit,
        -ball_vel_x,
        ball_vel_x,
    )

    return ball_x, ball_y, ball_vel_x, ball_vel_y


def enemy_step(state, step_counter, ball_y, ball_speed_y):
    # Skip movement every 8th step
    should_move = step_counter % 8 != 0

    # Calculate direction (-1 for up, 0 for stay, 1 for down)
    direction = jnp.sign(ball_y - state.enemy_y)

    # Calculate new position
    new_y = state.enemy_y + (direction * ENEMY_STEP_SIZE).astype(jnp.int32)
    # Return either new position or current position based on should_move
    return jax.lax.cond(
        should_move, lambda _: new_y, lambda _: state.enemy_y, operand=None
    )


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
        BALL_START_X.astype(jnp.int32),
        BALL_START_Y.astype(jnp.int32),
        ball_vel_x.astype(jnp.int32),
        ball_vel_y.astype(jnp.int32),
    )


class Game:
    def __init__(self, frameskip=0):
        self.frameskip = frameskip + 1

    def reset(self) -> State:
        """
        Resets the game state to the initial state.
        Returns the initial state and the reward (i.e. 0)
        """
        state = State(
            player_y=jnp.array(96).astype(jnp.int32),
            player_speed=jnp.array(0.0).astype(jnp.int32),
            ball_x=jnp.array(78).astype(jnp.int32),
            ball_y=jnp.array(115).astype(jnp.int32),
            enemy_y=jnp.array(115).astype(jnp.int32),
            enemy_speed=jnp.array(0.0).astype(jnp.int32),
            ball_vel_x=BALL_SPEED[0].astype(jnp.int32),
            ball_vel_y=BALL_SPEED[1].astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            enemy_score=jnp.array(0).astype(jnp.int32),
            step_counter=jnp.array(0).astype(jnp.int32),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            buffer=jnp.array(96).astype(jnp.int32),
        )
        return state, self._get_observation(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        # Step 1: Update player position and speed
        # only execute player step on even steps (base implementation only moves the player every second tick)
        new_player_y, player_speed_b, new_acceleration_counter = player_step(
            state.player_y, state.player_speed, state.acceleration_counter, action
        )

        new_player_y, player_speed, new_acceleration_counter = jax.lax.cond(
            state.step_counter % 2 == 0,
            lambda _: (new_player_y, player_speed_b, new_acceleration_counter),
            lambda _: (state.player_y, state.player_speed, state.acceleration_counter),
            operand=None,
        )

        buffer = jax.lax.cond(
            jax.lax.eq(state.buffer, state.player_y),
            lambda _: new_player_y,
            lambda _: state.buffer,
            operand=None,
        )
        player_y = state.buffer

        enemy_y = enemy_step(state, state.step_counter, state.ball_y, state.ball_y)

        # Step 2: Update ball position and velocity
        ball_x, ball_y, ball_vel_x, ball_vel_y = ball_step(state, action)

        # Step 3: Score and goal detection
        player_goal = ball_x < 4
        enemy_goal = ball_x > 156
        ball_reset = jnp.logical_or(enemy_goal, player_goal)

        # Step 4: Update scores
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

        # Step 5: Reset ball if goal was scored
        current_values = (
            ball_x.astype(jnp.int32),
            ball_y.astype(jnp.int32),
            ball_vel_x.astype(jnp.int32),
            ball_vel_y.astype(jnp.int32),
        )
        ball_x_final, ball_y_final, ball_vel_x_final, ball_vel_y_final = jax.lax.cond(
            ball_reset,
            lambda x: _reset_ball_after_goal((state, enemy_goal)),
            lambda x: x,
            operand=current_values,
        )

        # Step 6: Update step counter for game freeze after goal
        step_counter = jax.lax.cond(
            ball_reset,
            lambda s: jnp.array(0),
            lambda s: s + 1,
            operand=state.step_counter,
        )

        # Step 7: Update enemy position and speed

        # Step 8: Reset enemy position on goal
        enemy_y_final = jax.lax.cond(
            ball_reset,
            lambda s: BALL_START_Y.astype(jnp.int32),
            lambda s: enemy_y.astype(jnp.int32),
            operand=None,
        )

        # Step 9: Handle ball position during game freeze
        ball_x_final = jax.lax.cond(
            step_counter < 60,
            lambda s: BALL_START_X.astype(jnp.int32),
            lambda s: s,
            operand=ball_x_final,
        )
        ball_y_final = jax.lax.cond(
            step_counter < 60,
            lambda s: BALL_START_Y.astype(jnp.int32),
            lambda s: s,
            operand=ball_y_final,
        )

        new_state = State(
            player_y=player_y,
            player_speed=player_speed,
            ball_x=ball_x_final,
            ball_y=ball_y_final,
            enemy_y=enemy_y_final,
            enemy_speed=0,
            ball_vel_x=ball_vel_x_final,
            ball_vel_y=ball_vel_y_final,
            player_score=player_score,
            enemy_score=enemy_score,
            step_counter=step_counter,
            acceleration_counter=new_acceleration_counter,
            buffer=buffer,
        )

        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return new_state, obs, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: State):
        # create player
        player = EntityPosition(
            x=jnp.array(PLAYER_X),
            y=state.player_y,
            width=jnp.array(PLAYER_SIZE[0]),
            height=jnp.array(PLAYER_SIZE[1]),
        )

        # create enemy
        enemy = EntityPosition(
            x=jnp.array(ENEMY_X),
            y=state.enemy_y,
            width=jnp.array(ENEMY_SIZE[0]),
            height=jnp.array(ENEMY_SIZE[1]),
        )

        ball = EntityPosition(
            x=state.ball_x,
            y=state.ball_y,
            width=jnp.array(BALL_SIZE[0]),
            height=jnp.array(BALL_SIZE[1]),
        )
        return PongObservation(player=player, enemy=enemy, ball=ball, score_player=state.player_score, score_enemy=state.enemy_score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: State) -> PongInfo:
        return PongInfo(time=state.step_counter)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: State, state: State):
        return (state.player_score - state.enemy_score) - (previous_state.player_score - previous_state.enemy_score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: State) -> bool:
        return jnp.logical_or(jnp.greater_equal(state.player_score, 20), jnp.greater_equal(state.enemy_score, 20))


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

    def get_rgb_img(self, state: jnp.ndarray) -> np.ndarray:
        canvas = self.jax_rendering(self.convert_jax_arr(state))
        return np.array(canvas)

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
            canvas, state["player_score"], position=(116, 1), color=PLAYER_COLOR
        )
        self.draw_score(
            canvas, state["enemy_score"], position=(36, 1), color=ENEMY_COLOR
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
    game = Game(frameskip=1)

    # create a Renderer instance
    renderer = Renderer(STATE_TRANSLATOR)

    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_state, obs = jitted_reset()
    # Run the game until the user quits
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
                        curr_state, obs, reward, done, info = jitted_step(curr_state, action)

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_state, obs, reward, done, info = jitted_step(curr_state, action)

        renderer.display(screen, curr_state)
        counter += 1
        clock.tick(60)

    # Quit Pygame
    pygame.quit()
