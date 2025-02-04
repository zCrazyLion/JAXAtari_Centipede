from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp
import chex
import pygame

# Constants for game environment
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Colors from the original game
BACKGROUND_COLOR = (0, 0, 0)  # Black background
PLAYER_COLOR = (200, 72, 72)  # Red paddle
BALL_COLOR = (200, 72, 72)  # Red ball
WALL_COLOR = (142, 142, 142)  # Grey walls

# Block colors in order from top to bottom
BLOCK_COLORS = [
    (200, 72, 72),  # Red
    (198, 108, 58),  # Light Red
    (180, 122, 48),  # Orange
    (162, 162, 42),  # Yellow
    (72, 160, 72),  # Green
    (66, 72, 200),  # Blue
]

# Object sizes and positions
PLAYER_SIZE = (16, 4)  # Width, Height of paddle
BALL_SIZE = (2, 4)  # Width, Height of ball
BLOCK_SIZE = (8, 6)  # Width, Height of blocks

# Wall positions and sizes
WALL_TOP_Y = 24
WALL_TOP_HEIGHT = 10
WALL_SIDE_WIDTH = 8

# Initial positions
PLAYER_START_X = 99
PLAYER_START_Y = 189
BALL_START_X = 127  # TODO add other starting positions and directions
BALL_START_Y = 113

# Game boundaries (adjusted for wall width)
PLAYER_X_MIN = WALL_SIDE_WIDTH
PLAYER_X_MAX = 160 - PLAYER_SIZE[0]
PLAYER_MAX_SPEED = 6
PLAYER_ACCELERATION = jnp.array([3, 2, -1, 1, 1])
PLAYER_WALL_ACCELERATION = jnp.array([1, 2, 1, 1, 1])

# Block layout
BLOCKS_PER_ROW = 18
NUM_ROWS = 6
BLOCK_START_Y = 57  # Starting Y position of first row
BLOCK_START_X = 8  # Starting X position of blocks

# Ball speed TODO adjust
BALL_SPEED = jnp.array([-1, 1])

NUM_LIVES = 5


# Game state container
class State(NamedTuple):
    player_x: chex.Array
    player_speed: chex.Array
    ball_x: chex.Array
    ball_y: chex.Array
    ball_vel_x: chex.Array
    ball_vel_y: chex.Array
    blocks: chex.Array
    score: chex.Array
    lives: chex.Array
    step_counter: chex.Array
    acceleration_counter: chex.Array
    game_started: chex.Array


# Actions
NOOP = 0
FIRE = 1
RIGHT = 2
LEFT = 3


def get_human_action() -> chex.Array:
    """Records keyboard input and returns the corresponding action."""
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a]:
        return jnp.array(LEFT)
    elif keys[pygame.K_d]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_SPACE]:
        return jnp.array(FIRE)
    else:
        return jnp.array(NOOP)


def player_step(state_player_x: chex.Array,
                state_player_speed: chex.Array,
                acceleration_counter: chex.Array,
                action: chex.Array) -> (chex.Array, chex.Array, chex.Array):
    """Updates the player position based on the action."""
    left = (action == LEFT)
    right = (action == RIGHT)

    # Check if the paddle is touching the left or right wall.
    touches_wall = jnp.logical_or(state_player_x <= PLAYER_X_MIN, state_player_x >= PLAYER_X_MAX)

    # Get the acceleration schedule based on whether the paddle is at a wall.
    # If touching a wall, use PLAYER_WALL_ACCELERATION, otherwise use PLAYER_ACCELERATION.
    acceleration = jax.lax.cond(
        touches_wall,
        lambda _: PLAYER_WALL_ACCELERATION[acceleration_counter],
        lambda _: PLAYER_ACCELERATION[acceleration_counter],
        operand=None
    )

    # Apply deceleration if no button is pressed or if the paddle touches a wall.
    player_speed = jax.lax.cond(
        jnp.logical_or(jnp.logical_not(jnp.logical_or(left, right)), touches_wall),
        lambda s: jnp.round(s / 2).astype(jnp.int32),
        lambda s: s,
        operand=state_player_speed,
    )

    # If the paddle is moving to the right but the player pressed left, reset speed.
    direction_change_left = jnp.logical_and(left, state_player_speed > 0)
    player_speed = jax.lax.cond(
        direction_change_left,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )

    # Likewise, if moving left but the player pressed right, reset speed.
    direction_change_right = jnp.logical_and(right, state_player_speed < 0)
    player_speed = jax.lax.cond(
        direction_change_right,
        lambda s: 0,
        lambda s: s,
        operand=player_speed,
    )

    # If a direction change occurred, reset the acceleration counter.
    direction_change = jnp.logical_or(direction_change_left, direction_change_right)
    acceleration_counter = jax.lax.cond(
        direction_change,
        lambda _: 0,
        lambda s: s,
        operand=acceleration_counter,
    )

    # Apply acceleration for the pressed direction:
    # For left, we subtract the acceleration (making speed more negative), clamped to -MAX_SPEED.
    player_speed = jax.lax.cond(
        left,
        lambda s: jnp.maximum(s - acceleration, -PLAYER_MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    # For right, we add the acceleration (making speed more positive), clamped to MAX_SPEED.
    player_speed = jax.lax.cond(
        right,
        lambda s: jnp.minimum(s + acceleration, PLAYER_MAX_SPEED),
        lambda s: s,
        operand=player_speed,
    )

    # Update the acceleration counter: increment if a directional button is held, otherwise reset.
    new_acceleration_counter = jax.lax.cond(
        jnp.logical_or(left, right),
        lambda s: jnp.minimum(s + 1, PLAYER_ACCELERATION.size - 1),
        lambda s: 0,
        operand=acceleration_counter,
    )

    # Update the paddle's horizontal position and clamp it within the game boundaries.
    player_x = jnp.clip(state_player_x + player_speed, PLAYER_X_MIN, PLAYER_X_MAX)

    return player_x, player_speed, new_acceleration_counter


def ball_step(state, game_started, player_x):
    """Updates the ball's position, handles wall collisions, and paddle bounces."""
    ball_x = jnp.where(game_started, state.ball_x + state.ball_vel_x, BALL_START_X)
    ball_y = jnp.where(game_started, state.ball_y + state.ball_vel_y, BALL_START_Y)

    # Ball collision with side walls
    wall_left = ball_x <= WALL_SIDE_WIDTH
    wall_right = ball_x >= 160 - WALL_SIDE_WIDTH - BALL_SIZE[0]
    ball_x = jnp.where(
        wall_left,
        WALL_SIDE_WIDTH,
        jnp.where(wall_right, 160 - WALL_SIDE_WIDTH - BALL_SIZE[0], ball_x),
    )
    ball_vel_x = jnp.where(wall_left | wall_right, -state.ball_vel_x, state.ball_vel_x)

    # Ball collision with top wall
    wall_top = ball_y <= WALL_TOP_Y + WALL_TOP_HEIGHT
    ball_y = jnp.where(wall_top, WALL_TOP_Y + WALL_TOP_HEIGHT, ball_y)
    ball_vel_y = jnp.where(wall_top, -state.ball_vel_y, state.ball_vel_y)

    # Paddle collision
    paddle_hit = jnp.logical_and(
        ball_y + BALL_SIZE[1] >= PLAYER_START_Y,
        jnp.logical_and(
            ball_x + BALL_SIZE[0] >= player_x, ball_x <= player_x + PLAYER_SIZE[0]
        ),
    )

    section_width = PLAYER_SIZE[0] / 5  # Divide paddle into 5 sections
    hit_section = jnp.where(
        paddle_hit, jnp.floor((ball_x - player_x) / section_width).astype(jnp.int32), 0
    )

    # Adjust ball velocity based on hit section
    ball_vel_x = jnp.where(
        paddle_hit,
        jnp.where(
            hit_section == 0,
            -2,  # Leftmost section
            jnp.where(
                hit_section == 1,
                -1,  # Second section from left
                jnp.where(
                    hit_section == 3,
                    1,  # Second section from right
                    jnp.where(hit_section == 4, 2, ball_vel_x),
                ),
            ),
        ),  # Rightmost section
        ball_vel_x,
    )

    ball_vel_y = jnp.where(
        paddle_hit, -jnp.abs(ball_vel_y), ball_vel_y
    )  # Always bounce upwards

    return ball_x, ball_y, ball_vel_x, ball_vel_y


def check_block_collision(
    state, ball_x, ball_y, ball_vel_x, ball_vel_y
):  # TODO better bouncing and ball speed
    """Checks for block collisions and updates the state."""

    def collision_logic(carry, block_idx):
        blocks, score, ball_x, ball_y, ball_vel_x, ball_vel_y = carry
        row, col = block_idx // BLOCKS_PER_ROW, block_idx % BLOCKS_PER_ROW

        block_x = BLOCK_START_X + col * BLOCK_SIZE[0]
        block_y = BLOCK_START_Y + row * BLOCK_SIZE[1]

        block_hit = jnp.logical_and(
            blocks[row, col] == 1,
            jnp.logical_and(
                jnp.logical_and(
                    ball_x <= block_x + BLOCK_SIZE[0], ball_x + BALL_SIZE[0] >= block_x
                ),
                jnp.logical_and(
                    ball_y <= block_y + BLOCK_SIZE[1], ball_y + BALL_SIZE[1] >= block_y
                ),
            ),
        )

        # Collision side detection
        dx = (ball_x + BALL_SIZE[0] / 2) - (block_x + BLOCK_SIZE[0] / 2)
        dy = (ball_y + BALL_SIZE[1] / 2) - (block_y + BLOCK_SIZE[1] / 2)
        is_horizontal = jnp.abs(dx / BLOCK_SIZE[0]) > jnp.abs(dy / BLOCK_SIZE[1])

        ball_vel_x = jnp.where(block_hit & is_horizontal, -ball_vel_x, ball_vel_x)
        ball_vel_y = jnp.where(block_hit & ~is_horizontal, -ball_vel_y, ball_vel_y)

        # Update block visibility and score
        blocks = blocks.at[row, col].set(jnp.where(block_hit, 0, blocks[row, col]))

        # Update score based on row
        points = jnp.where(row >= 4, 1, jnp.where(row >= 2, 4, 7))
        score = score + jnp.where(block_hit, points, 0)

        return (blocks, score, ball_x, ball_y, ball_vel_x, ball_vel_y), None

    (new_blocks, new_score, ball_x, ball_y, ball_vel_x, ball_vel_y), _ = jax.lax.scan(
        collision_logic,
        (state.blocks, state.score, ball_x, ball_y, ball_vel_x, ball_vel_y),
        jnp.arange(NUM_ROWS * BLOCKS_PER_ROW),
    )

    return new_blocks, new_score, ball_x, ball_y, ball_vel_x, ball_vel_y


class Game:
    def __init__(self, frameskip=1):
        self.frameskip = frameskip

    def reset(self) -> State:
        """Initialize game state"""
        return State(
            player_x=jnp.array(PLAYER_START_X),
            player_speed=jnp.array(0),
            ball_x=jnp.array(BALL_START_X),
            ball_y=jnp.array(BALL_START_Y),
            ball_vel_x=BALL_SPEED[0],
            ball_vel_y=BALL_SPEED[1],
            blocks=jnp.ones((NUM_ROWS, BLOCKS_PER_ROW), dtype=jnp.int32),
            score=jnp.array(0),
            lives=jnp.array(NUM_LIVES),
            step_counter=jnp.array(0),
            acceleration_counter=jnp.array(0).astype(jnp.int32),
            game_started=jnp.array(0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        new_player_x, new_paddle_v, new_acceleration_counter = player_step(state.player_x, state.player_speed, state.acceleration_counter, action)

        game_started = jnp.logical_or(state.game_started, action == FIRE)

        # Update ball, check collisions, etc., as before, but now pass new_player_x
        ball_x, ball_y, ball_vel_x, ball_vel_y = ball_step(state, game_started, new_player_x)
        new_blocks, new_score, ball_x, ball_y, ball_vel_x, ball_vel_y = (
            check_block_collision(state, ball_x, ball_y, ball_vel_x, ball_vel_y)
        )

        # Handle life loss, etc.
        life_lost = ball_y >= WINDOW_HEIGHT // 3
        ball_x = jnp.where(life_lost, new_player_x + 7, ball_x)
        ball_y = jnp.where(life_lost, BALL_START_Y, ball_y)
        ball_vel_x = jnp.where(life_lost, BALL_SPEED[0], ball_vel_x)
        ball_vel_y = jnp.where(life_lost, BALL_SPEED[1], ball_vel_y)
        game_started = jnp.where(life_lost, jnp.array(0), game_started)
        new_lives = jnp.where(life_lost, state.lives - 1, state.lives)

        return State(
            player_x=new_player_x,
            player_speed=new_paddle_v,
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vel_x=ball_vel_x,
            ball_vel_y=ball_vel_y,
            blocks=new_blocks,
            score=new_score,
            lives=new_lives,
            step_counter=state.step_counter + 1,
            acceleration_counter=new_acceleration_counter,
            game_started=game_started,
        )


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Breakout")
        self.clock = pygame.time.Clock()

    def render(self, state: State):
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)

        # Draw walls
        # Top wall
        pygame.draw.rect(
            self.screen,
            WALL_COLOR,
            (0, WALL_TOP_Y * 3, WINDOW_WIDTH, WALL_TOP_HEIGHT * 3),
        )

        # Left wall
        pygame.draw.rect(
            self.screen,
            WALL_COLOR,
            (0, WALL_TOP_Y * 3, WALL_SIDE_WIDTH * 3, WINDOW_HEIGHT),
        )

        # Right wall
        pygame.draw.rect(
            self.screen,
            WALL_COLOR,
            (
                WINDOW_WIDTH - WALL_SIDE_WIDTH * 3,
                WALL_TOP_Y * 3,
                WALL_SIDE_WIDTH * 3,
                WINDOW_HEIGHT,
            ),
        )

        # Draw blocks
        for row in range(NUM_ROWS):
            for col in range(BLOCKS_PER_ROW):
                if state.blocks[row, col] == 1:
                    block_rect = pygame.Rect(
                        (BLOCK_START_X + col * BLOCK_SIZE[0]) * 3,
                        (BLOCK_START_Y + row * BLOCK_SIZE[1]) * 3,
                        BLOCK_SIZE[0] * 3,
                        BLOCK_SIZE[1] * 3,
                    )
                    pygame.draw.rect(self.screen, BLOCK_COLORS[row], block_rect)

        # Draw player paddle
        player_rect = pygame.Rect(
            int(state.player_x) * 3,
            PLAYER_START_Y * 3,
            PLAYER_SIZE[0] * 3,
            PLAYER_SIZE[1] * 3,
        )
        pygame.draw.rect(self.screen, PLAYER_COLOR, player_rect)

        # Draw ball only if the game has started
        if state.game_started:
            ball_rect = pygame.Rect(
                int(state.ball_x) * 3,
                int(state.ball_y) * 3,
                BALL_SIZE[0] * 3,
                BALL_SIZE[1] * 3,
            )
            pygame.draw.rect(self.screen, BALL_COLOR, ball_rect)

        # Draw score and lives
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {state.score}", True, (255, 255, 255))
        lives_text = font.render(f"Lives: {state.lives}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (WINDOW_WIDTH - 100, 10))

        pygame.display.flip()


if __name__ == "__main__":
    # Initialize game and renderer
    game = Game(frameskip=1)
    renderer = Renderer()

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_state = jitted_reset()

    # Game loop
    running = True
    frameskip = game.frameskip
    counter = 1

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if counter % frameskip == 0:
            action = get_human_action()
            curr_state = jitted_step(curr_state, action)

        # Check for game over
        if curr_state.lives < 0:
            running = False
        else:
            renderer.render(curr_state)

        counter += 1
        renderer.clock.tick(60)

    pygame.quit()
