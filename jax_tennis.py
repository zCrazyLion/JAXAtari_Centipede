from functools import partial
from typing import NamedTuple, Tuple, Any
import jax
import jax.numpy as jnp
import chex
import numpy as np
import pygame


# Action constants (placeholders - to be defined according to ALE)
NOOP = 0
FIRE = 1
UP = 2
DOWN = 3
LEFT = 4
RIGHT = 5

# Game constants (placeholders - to be adjusted according to ALE)
COURT_WIDTH = 160
COURT_HEIGHT = 210
PLAYER_START_X = 71
PLAYER_START_Y = 24
ENEMY_START_X = 71  # this is where he starts, but he instantly moves to:
ENEMY_START_Y = 159
BALL_START_X = 75  # taken from the RAM extraction script (initial ram 77 - 2)
BALL_START_Y = 44  # taken from the RAM extraction script (189 - initial ram 145)
BALL_START_Z = 7  # of course there is no real z, but using the shadow it is suggested that there is a z
PLAYER_WIDTH = 13
PLAYER_HEIGHT = 23
BALL_SIZE = 2

WAIT_AFTER_GOAL = 0  # number of ticks that are waited after a goal was scored

# game constrains (i.e. the net)
NET_RANGE = (98, 113)

# TODO: define the constraints of everyone (player, ball, enemy) according to the base implementation


# TODO: due to the movement properties of the ball in Tennis, the velocity values should probably represent how frequent ticks are skipped..
# Define state container
class State(NamedTuple):
    player_x: chex.Array  # Player x position
    player_y: chex.Array  # Player y position
    enemy_x: chex.Array  # Enemy x position
    enemy_y: chex.Array  # Enemy y position
    ball_x: chex.Array  # Ball x position
    ball_y: chex.Array  # Ball y position
    ball_z: chex.Array  # Ball height/shadow
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    ball_vel_z: chex.Array
    player_score: chex.Array
    enemy_score: chex.Array
    serving: chex.Array  # boolean for serve state
    current_tick: chex.Array


# State positions for translation
STATE_TRANSLATOR = {
    0: "player_x",
    1: "player_y",
    2: "enemy_x",
    3: "enemy_y",
    4: "ball_x",
    5: "ball_y",
    6: "ball_z",
    7: "ball_vel_x",
    8: "ball_vel_y",
    9: "ball_vel_z",
    10: "player_score",
    11: "enemy_score",
    12: "serving",
    13: "current_tick",
}


def get_human_action() -> chex.Array:
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    elif keys[pygame.K_w]:
        return jnp.array(UP)
    elif keys[pygame.K_s]:
        return jnp.array(DOWN)
    else:
        return jnp.array(NOOP)


@partial(jax.jit, static_argnums=())
def update_z_position(z_pos: chex.Array) -> chex.Array:
    """Updates the z-position following the exact pattern from the game data.

    The z-value follows this sequence:
    - At start or after reset: Begins at 7
    - Initial fast rise (7-14): Increments by 1
    - Mid-rise (15-29): Variable increments of 1
    - Peak approach (30-38): Slower rise
    - Peak (38-39): Alternates between 38 and 39
    - Reset to 0: After reaching peak, resets and starts cycle

    Args:
        z_pos: Current z position value

    Returns:
        chex.Array: New z position value
    """
    # Constants
    PEAK_Z = jnp.array(38, dtype=jnp.int32)
    MIN_Z = jnp.array(0, dtype=jnp.int32)

    # Define the initial rise sequence for precise control
    initial_sequence = jnp.array(
        [
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ],
        dtype=jnp.int32,
    )

    # Handle the rise phase
    def rising_phase(z):
        # If in initial sequence range, move to next value
        in_initial = z < initial_sequence[-1]
        initial_idx = jnp.sum(initial_sequence <= z)
        next_initial = jnp.where(
            initial_idx < len(initial_sequence), initial_sequence[initial_idx], z + 1
        )

        # Normal rise pattern after initial sequence
        normal_rise = z + 1

        # Combine patterns
        return jnp.where(in_initial, next_initial, jnp.minimum(normal_rise, PEAK_Z))

    # Handle peak alternation
    def handle_peak(z):
        at_peak = z >= PEAK_Z
        return jnp.where(at_peak, jnp.where(z == PEAK_Z, PEAK_Z + 1, PEAK_Z), z)

    # Reset logic
    def handle_reset(z):
        should_reset = z >= PEAK_Z + 1
        return jnp.where(should_reset, MIN_Z, z)

    # Apply the update phases
    new_z = rising_phase(z_pos)
    new_z = handle_peak(new_z)
    new_z = handle_reset(new_z)

    return new_z


@partial(jax.jit, static_argnums=())
def ball_step(
    ball_x: chex.Array,
    ball_y: chex.Array,
    ball_z: chex.Array,
    ball_vel_x: chex.Array,
    ball_vel_y: chex.Array,
    ball_vel_z: chex.Array,
    serving: chex.Array,
    current_tick: chex.Array,
    top_x: chex.Array,
    top_y: chex.Array,
    bot_x: chex.Array,
    bot_y: chex.Array,
    collision: chex.Array,
    action: chex.Array,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """
    Updates ball position and velocity based on current state and action.
    All calculations use integer math to match Atari 2600 capabilities.

    Args:
        ball_x: Current ball x position
        ball_y: Current ball y position
        ball_z: Current ball z position (height/shadow)
        ball_vel_x: Current ball x velocity
        ball_vel_y: Current ball y velocity
        ball_vel_z: Current ball z velocity
        serving: Whether ball is currently being served
        current_tick: Current game tick
        top_x: X position of top racket
        top_y: Y position of top racket
        bot_x: X position of bottom racket
        bot_y: Y position of bottom racket
        collision: Whether ball collided with racket this frame
        action: Current player action

    Returns:
        Tuple containing new:
        - ball_x: Ball x position (int32)
        - ball_y: Ball y position (int32)
        - ball_z: Ball height/shadow distance (int32)
        - ball_vel_x: Ball x velocity (int32)
        - ball_vel_y: Ball y velocity (int32)
        - ball_vel_z: Ball vertical velocity (int32)
        - serving: Updated serving state
    """
    BASE_Y_VEL = jnp.array(2, dtype=jnp.int32)

    # Update z position first
    new_z = update_z_position(ball_z)

    # Y movement sequence synchronized with z movement
    y_step = jnp.where(
        ball_vel_y > 0,
        jnp.array(2, dtype=jnp.int32),  # Moving down
        jnp.array(-2, dtype=jnp.int32),  # Moving up
    )

    # Only apply y movement if not serving and ball is in motion
    ball_y_updated = jnp.where(ball_vel_y != 0, ball_y + y_step, ball_y)

    # Handle net crossing
    net_y = (NET_RANGE[0] + NET_RANGE[1]) // 2
    approaching_net = jnp.logical_and(
        ball_y_updated >= NET_RANGE[0], ball_y_updated <= NET_RANGE[1]
    )

    teleport_dist = jnp.where(
        ball_y_updated < net_y,
        jnp.array(40, dtype=jnp.int32),
        jnp.array(-40, dtype=jnp.int32),
    )

    ball_y = jnp.where(approaching_net, ball_y_updated + teleport_dist, ball_y_updated)

    # Handle hits
    def compute_hit_response(curr_vel_x, curr_vel_y):
        racket_center = jnp.where(
            ball_y < net_y, top_x + PLAYER_WIDTH // 2, bot_x + PLAYER_WIDTH // 2
        )

        hit_offset = ball_x - racket_center

        new_x_vel = jnp.where(
            jnp.abs(hit_offset) > PLAYER_WIDTH // 4,
            jnp.sign(hit_offset),
            jnp.where(
                jnp.abs(hit_offset) > PLAYER_WIDTH // 8,
                jnp.sign(hit_offset),
                jnp.array(0, dtype=jnp.int32),
            ),
        )

        new_y_vel = -jnp.sign(curr_vel_y) * BASE_Y_VEL
        ball_y_adj = jnp.where(ball_y < net_y, ball_y + 10, ball_y)

        return new_x_vel, new_y_vel, ball_x, ball_y_adj

    # Update velocities and position on hit
    new_vel_x, new_vel_y, new_ball_x, new_ball_y = jax.lax.cond(
        collision,
        lambda _: compute_hit_response(ball_vel_x, ball_vel_y),
        lambda _: (ball_vel_x, ball_vel_y, ball_x, ball_y),
        None,
    )

    # Normal movement
    final_x = jnp.where(collision, new_ball_x, ball_x + new_vel_x)
    final_y = jnp.where(collision, new_ball_y, ball_y + new_vel_y)

    # Serve logic
    def serve_motion():
        # Track if FIRE was pressed to start serve
        serve_started = action == FIRE

        # Initial serve velocity on FIRE press
        initial_serve_vel = jnp.where(
            serve_started & (ball_vel_y == 0),
            jnp.array(2, dtype=jnp.int32),  # Initial y velocity
            jnp.array(0, dtype=jnp.int32),
        )

        # Pre-serve ball movement (small up/down bouncing)
        t = current_tick % 40
        serve_y_step = jnp.where(
            t < 20,
            jnp.array(-2, dtype=jnp.int32),  # Moving up
            jnp.array(2, dtype=jnp.int32),  # Moving down
        )

        # Only apply serve bounce if not started
        new_serve_y = jnp.where(
            serve_started,
            ball_y + jnp.array(-2, dtype=jnp.int32),  # Start moving up when served
            ball_y + serve_y_step,  # Continue bounce pattern
        )

        serve_hit = collision & (initial_serve_vel != 0)

        def handle_serve_collision():
            new_x_vel, new_y_vel, hit_x, hit_y = compute_hit_response(
                0, initial_serve_vel
            )
            return hit_x, hit_y, new_z, new_x_vel, new_y_vel, 0, False

        return jax.lax.cond(
            serve_hit,
            lambda _: handle_serve_collision(),
            lambda _: (
                ball_x,
                new_serve_y,
                new_z,
                0,
                initial_serve_vel,
                0,
                ~serve_started,
            ),
            None,
        )

    # Choose between serve and normal motion
    return jax.lax.cond(
        serving,
        lambda _: serve_motion(),
        lambda _: (final_x, final_y, new_z, new_vel_x, new_vel_y, ball_vel_z, serving),
        None,
    )


def player_step(
    state_player_x: chex.Array, state_player_y: chex.Array, action: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """
    Updates player position based on current position and action.

    Args:
        state_player_x: Current x coordinate of player
        state_player_y: Current y coordinate of player
        action: Current player action

    Returns:
        Tuple containing:
            chex.Array: New player x position
            chex.Array: New player y position
    """

    # TODO: adjust the borders of the game according to base implementation
    # check if the player is trying to move left
    player_x = jnp.where(
        jnp.logical_and(action == LEFT, state_player_x > 0),
        state_player_x - 1,
        state_player_x,
    )

    # check if the player is trying to move right
    player_x = jnp.where(
        jnp.logical_and(action == RIGHT, state_player_x < COURT_WIDTH - 13),
        state_player_x + 1,
        player_x,
    )

    # check if the player is trying to move up
    player_y = jnp.where(
        jnp.logical_and(action == UP, state_player_y > 0),
        state_player_y - 1,
        state_player_y,
    )

    # check if the player is trying to move down
    player_y = jnp.where(
        jnp.logical_and(action == DOWN, state_player_y < COURT_HEIGHT - 23),
        state_player_y + 1,
        player_y,
    )

    return player_x, player_y


def enemy_step(
    state_enemy_x: chex.Array,
    state_enemy_y: chex.Array,
    ball_x: chex.Array,
    ball_y: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """
    Updates enemy position based on current position and ball position.

    Args:
        state_enemy_x: Current x coordinate of enemy
        state_enemy_y: Current y coordinate of enemy
        ball_x: Current x coordinate of ball
        ball_y: Current y coordinate of ball

    Returns:
        Tuple containing:
            chex.Array: New enemy x position
            chex.Array: New enemy y position
    """

    # TODO: basic implementation for now

    # move 1 towards the ball in x
    enemy_x = jnp.where(ball_x < state_enemy_x, state_enemy_x - 1, state_enemy_x)

    enemy_x = jnp.where(ball_x > state_enemy_x, state_enemy_x + 1, enemy_x)

    # for now dont move in y, I think there is some distance based movement happening

    return enemy_x, state_enemy_y


def check_scoring(state: State) -> Tuple[chex.Array, chex.Array, bool]:
    """
    Checks if a point was scored and updates the score accordingly.

    Args:
        state: Current game state containing ball and player positions and scores

    Returns:
        Tuple containing:
            chex.Array: Updated player score
            chex.Array: Updated enemy score
            bool: Whether a point was scored in this step
    """
    return state.player_score, state.enemy_score, False


def check_collision(
    player_x, player_y, ball_x, ball_y, ball_z, enemy_x, enemy_y
) -> chex.Array:
    """
    Checks if a collision occurred between the ball and the player or enemy.

    Args:
        player_x: Current x coordinate of player
        player_y: Current y coordinate of player
        ball_x: Current x coordinate of ball
        ball_y: Current y coordinate of ball
        ball_z: Current z coordinate of ball
        enemy_x: Current x coordinate of enemy
        enemy_y: Current y coordinate of enemy

    Returns:
        bool: Whether a collision occurred
        bool: Whether the z position of the ball is valid for the collision
    """
    # Correct detection: a collision is registered, when the ball and the racket have pixel overlap (with small margin)
    # Assumption for now: a collision is registered, when the ball is in the 50% of the player that are closer to the ball
    # The player is always turned towards the ball, therefore we can just check if the ball is intercepting with the player / enemy

    # Define valid hit zones
    TOP_VALID_Y = (24, 32)
    BOTTOM_VALID_Y = (138, 154)
    TOP_VALID_Z = (16, 22)
    BOTTOM_VALID_Z = (0, 15)

    def check_hit_zone(
        y_pos: chex.Array, z_pos: chex.Array, valid_y: tuple, valid_z: tuple
    ) -> chex.Array:
        y_valid = jnp.logical_and(y_pos >= valid_y[0], y_pos <= valid_y[1])
        z_valid = jnp.logical_and(z_pos >= valid_z[0], z_pos <= valid_z[1])
        return jnp.logical_and(y_valid, z_valid)

    # Check overlaps
    player_overlap = jnp.logical_and(
        jnp.logical_and(
            ball_x < player_x + PLAYER_WIDTH, ball_x + BALL_SIZE > player_x
        ),
        jnp.logical_and(
            ball_y < player_y + PLAYER_HEIGHT, ball_y + BALL_SIZE > player_y
        ),
    )

    enemy_overlap = jnp.logical_and(
        jnp.logical_and(ball_x < enemy_x + PLAYER_WIDTH, ball_x + BALL_SIZE > enemy_x),
        jnp.logical_and(ball_y < enemy_y + PLAYER_HEIGHT, ball_y + BALL_SIZE > enemy_y),
    )

    # Combine with valid hit zones
    player_valid = check_hit_zone(ball_y, ball_z, TOP_VALID_Y, TOP_VALID_Z)
    enemy_valid = check_hit_zone(ball_y, ball_z, BOTTOM_VALID_Y, BOTTOM_VALID_Z)

    player_collision = jnp.logical_and(player_overlap, player_valid)
    enemy_collision = jnp.logical_and(enemy_overlap, enemy_valid)

    return jnp.logical_or(player_collision, enemy_collision)


def before_serve(state: State) -> chex.Array:
    """
    Plays the idle animation of the ball before serving.

    Args:
        state: Current game state

    Returns:
        chex.Array: Updated ball y position
    """

    # idle movement of the ball
    idle_movement = jnp.array(
        [
            0,
            -2,
            -2,
            -3,
            -2,
            -2,
            -2,
            -1,
            -2,
            -2,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            0,
            0,
            -1,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            1,
            2,
            2,
            2,
            3,
            2,
            2,
            0,
        ]
    )

    # Calculate tick using modulo
    tick = jnp.mod(state.current_tick - WAIT_AFTER_GOAL, idle_movement.shape[0])

    # calculate the position in the idle movement array
    movement = jnp.take(idle_movement, tick)

    # Update ball_y
    ball_y = state.ball_y + movement

    return ball_y


class Game:
    def __init__(self, frameskip=0):
        self.frameskip = frameskip

    def reset(self) -> State:
        """Resets game to initial state."""
        return State(
            player_x=jnp.array(PLAYER_START_X).astype(jnp.int32),
            player_y=jnp.array(PLAYER_START_Y).astype(jnp.int32),
            enemy_x=jnp.array(ENEMY_START_X).astype(jnp.int32),
            enemy_y=jnp.array(ENEMY_START_Y).astype(jnp.int32),
            ball_x=jnp.array(BALL_START_X).astype(jnp.int32),
            ball_y=jnp.array(BALL_START_Y).astype(jnp.int32),
            ball_z=jnp.array(BALL_START_Z).astype(jnp.int32),
            ball_vel_x=jnp.array(0).astype(jnp.int32),
            ball_vel_y=jnp.array(0).astype(jnp.int32),
            ball_vel_z=jnp.array(0).astype(jnp.int32),
            player_score=jnp.array(0).astype(jnp.int32),
            enemy_score=jnp.array(0).astype(jnp.int32),
            serving=jnp.array(1).astype(jnp.bool),  # boolean for serve state
            current_tick=jnp.array(0).astype(jnp.int32),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        """Executes one game step."""
        # Update player position
        player_x, player_y = player_step(state.player_x, state.player_y, action)

        # check if there was a collision
        collision = check_collision(
            player_x,
            player_y,
            state.ball_x,
            state.ball_y,
            state.ball_z,
            state.enemy_x,
            state.enemy_y,
        )

        # Update ball position and velocity (TODO: currently no side switching implemented)
        ball_x, ball_y, ball_z, ball_vel_x, ball_vel_y, ball_vel_z, serve = ball_step(
            state.ball_x,
            state.ball_y,
            state.ball_z,
            state.ball_vel_x,
            state.ball_vel_y,
            state.ball_vel_z,
            state.serving,
            state.current_tick,
            player_x,
            player_y,
            state.enemy_x,
            state.enemy_y,
            collision,
            action,
        )

        # Check scoring
        player_score, enemy_score, point_scored = check_scoring(state)

        enemy_x, enemy_y = enemy_step(state.enemy_x, state.enemy_y, ball_x, ball_y)

        # if nothing is happening, play the idle animation of the ball
        ball_y = jnp.where(serve, before_serve(state), ball_y)

        # if its serve, block the y movement of player and enemy
        player_y = jnp.where(serve, PLAYER_START_Y, player_y)

        # if its serve, block the y movement of player and enemy
        enemy_y = jnp.where(serve, ENEMY_START_Y, state.enemy_y)

        # if the game is freezed, return the current state
        calculated_state = State(
            player_x=player_x,
            player_y=player_y,
            enemy_x=enemy_x,
            enemy_y=enemy_y,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_z=ball_z,
            ball_vel_x=ball_vel_x,
            ball_vel_y=ball_vel_y,
            ball_vel_z=ball_vel_z,
            player_score=player_score,
            enemy_score=enemy_score,
            serving=serve,
            current_tick=state.current_tick + 1,
        )
        return jax.lax.cond(
            state.current_tick < WAIT_AFTER_GOAL,
            lambda: state._replace(current_tick=state.current_tick + 1),
            lambda: calculated_state,
        )


class Renderer:
    def __init__(self, jax_translator):
        self.translator = jax_translator

    def display(self, screen, state):
        """Displays the rendered game state using pygame."""
        canvas = self.jax_rendering(self.convert_jax_arr(state))
        canvas_np = np.array(canvas)
        canvas_np = np.flipud(np.rot90(canvas_np, k=1))
        pygame_surface = pygame.surfarray.make_surface(canvas_np)
        screen.blit(
            pygame.transform.scale(pygame_surface, (COURT_WIDTH * 3, COURT_HEIGHT * 3)),
            (0, 0),
        )
        pygame.display.flip()

    def get_rgb_img(self, state: jnp.ndarray) -> np.ndarray:
        canvas = self.jax_rendering(self.convert_jax_arr(state))
        return np.array(canvas)

    def convert_jax_arr(self, state: jnp.array) -> dict:
        """Converts JAX array to dict using the translator."""
        state_dict = {}
        for i in range(len(state)):
            state_dict[self.translator[i]] = state[i]
        return state_dict

    def jax_rendering(self, state: dict):
        """Renders the current state of the game matching the original Atari style."""
        # Create dark green background
        canvas = np.full((COURT_HEIGHT, COURT_WIDTH, 3), [0, 100, 0], dtype=np.uint8)

        # Court dimensions
        court_top = 40
        court_bottom = 180
        center_x = COURT_WIDTH // 2

        # Calculate trapezoid points for perspective
        top_width = 100  # Width at top of court
        bottom_width = 140  # Width at bottom of court
        top_left = center_x - top_width // 2
        top_right = center_x + top_width // 2
        bottom_left = center_x - bottom_width // 2
        bottom_right = center_x + bottom_width // 2

        # Draw court outline (white lines)
        # Top line
        canvas[court_top, top_left:top_right] = [255, 255, 255]
        # Bottom line
        canvas[court_bottom, bottom_left:bottom_right] = [255, 255, 255]

        # Draw side lines with perspective
        for y in range(court_top, court_bottom + 1):
            # Calculate x positions for left and right lines
            progress = (y - court_top) / (court_bottom - court_top)
            left_x = int(top_left + (bottom_left - top_left) * progress)
            right_x = int(top_right + (bottom_right - top_right) * progress)
            canvas[y, left_x] = [255, 255, 255]
            canvas[y, right_x] = [255, 255, 255]

        # Draw net (solid gray band)
        net_y = (NET_RANGE[0] + NET_RANGE[1]) // 2
        net_height = 15  # Adjust net height as needed
        for y in range(net_y - net_height // 2, net_y + net_height // 2):
            progress = (y - court_top) / (court_bottom - court_top)
            left_x = int(top_left + (bottom_left - top_left) * progress)
            right_x = int(top_right + (bottom_right - top_right) * progress)
            canvas[y, left_x:right_x] = [200, 200, 200]

        # Draw player and enemy (white rectangles)
        player_x = int(state["player_x"])
        player_y = int(state["player_y"])
        if 0 <= player_y < COURT_HEIGHT - 23 and 0 <= player_x < COURT_WIDTH - 13:
            canvas[player_y : player_y + 23, player_x : player_x + 13] = [255, 255, 255]

        enemy_x = int(state["enemy_x"])
        enemy_y = int(state["enemy_y"])
        if 0 <= enemy_y < COURT_HEIGHT - 23 and 0 <= enemy_x < COURT_WIDTH - 13:
            canvas[enemy_y : enemy_y + 23, enemy_x : enemy_x + 13] = [255, 255, 255]

        # Draw ball and shadow
        ball_x = int(state["ball_x"])
        ball_y = int(state["ball_y"])
        ball_z = int(state["ball_z"])

        if 0 <= ball_y < COURT_HEIGHT - 2 and 0 <= ball_x < COURT_WIDTH - 2:
            # Draw ball shadow (gray square on court surface)
            canvas[ball_y : ball_y + 2, ball_x : ball_x + 2] = [100, 100, 100]

            # Draw ball with height offset based on z value
            ball_height_offset = ball_z // 2
            if ball_y - ball_height_offset >= 0:
                canvas[
                    ball_y - ball_height_offset : ball_y - ball_height_offset + 2,
                    ball_x : ball_x + 2,
                ] = [255, 255, 255]

        # Draw scores as simple squares in top corners
        # Player score (right)
        score_size = 8
        canvas[
            10 : 10 + score_size, COURT_WIDTH - 20 : COURT_WIDTH - 20 + score_size
        ] = [255, 255, 255]

        # Enemy score (left)
        canvas[10 : 10 + score_size, 20 : 20 + score_size] = [255, 255, 255]

        return jnp.array(canvas)

    def draw_score(self, canvas, score, position):
        """Draw score with improved visibility."""
        x, y = position
        score_str = str(score)
        digit_color = [255, 255, 255]  # White color for score

        # Draw background box for score
        box_padding = 2
        box_width = len(score_str) * 8 + box_padding * 2
        box_height = 12
        canvas[y - box_padding : y + box_height, x - box_padding : x + box_width] = [
            0,
            100,
            0,
        ]  # Dark green background

        # Draw digits
        for digit in score_str:
            # Simple digit rendering with improved spacing
            digit_int = int(digit)
            canvas[y : y + 8, x : x + 6] = digit_color
            x += 8  # Increased spacing between digits

    def draw_net(self, canvas):
        # Draw net (for simplicity it spans the whole x, and the y is the NET RANGE)
        canvas[NET_RANGE[0] : NET_RANGE[1], :] = [255, 255, 255]


# TODO: pull out the game loop into a main function that wraps all the different games
if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((COURT_WIDTH * 3, COURT_HEIGHT * 3))
    pygame.display.set_caption("Tennis Game")
    clock = pygame.time.Clock()

    # Create game instance
    game = Game(frameskip=1)

    # Initialize renderer
    renderer = Renderer(STATE_TRANSLATOR)

    # JIT compile main functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    # Main game loop structure
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
                        print("ball_x: " + str(curr_state.ball_x))
                        print("ball_y: " + str(curr_state.ball_y))
                        print("ball_z: " + str(curr_state.ball_z))

        if not frame_by_frame:
            if counter % frameskip == 0:
                # Get action (to be implemented with proper controls)
                action = get_human_action()
                curr_state: State = jitted_step(curr_state, action)

        renderer.display(screen, curr_state)

        counter += 1
        clock.tick(60)

    pygame.quit()
