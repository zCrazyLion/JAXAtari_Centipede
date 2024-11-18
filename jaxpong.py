import jax.numpy as jnp
import numpy as np
import pygame
from numbers_impl import digits

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

def get_human_action():
    """
    Records if UP or DOWN is being pressed and returns the corresponding action.

    Returns:
        action: int, action taken by the player (LEFT, RIGHT, FIRE, LEFTFIRE, RIGHTFIRE, NOOP).
    """
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] and keys[pygame.K_SPACE]:
        return LEFTFIRE
    elif keys[pygame.K_d] and keys[pygame.K_SPACE]:
        return RIGHTFIRE
    elif keys[pygame.K_a]:
        return LEFT
    elif keys[pygame.K_d]:
        return RIGHT
    elif keys[pygame.K_SPACE]:
        return FIRE
    else:
        return NOOP

class Game:
    def __init__(self):
        # Initialize game state
        self.player_y = 96
        self.player_speed = 0.0
        self.ball_x = 78
        self.ball_y = 115
        self.enemy_y = 115
        self.enemy_speed = 0.0
        self.ball_vel_x = BALL_SPEED[0]
        self.ball_vel_y = BALL_SPEED[1]
        self.player_score = 0
        self.enemy_score = 0
        self.step_counter = 0

    def player_step(self, action):
        """
        Updates the player's speed and position based on the action.
        """
        if action == LEFT:
            self.player_speed -= PLAYER_ACCELERATION
        elif action == RIGHT:
            self.player_speed += PLAYER_ACCELERATION
        else:
            self.player_speed *= 0.9  # Apply friction when no action is taken

        # Limit player speed to the maximum allowed value
        self.player_speed = jnp.clip(self.player_speed, -PLAYER_MAX_SPEED, PLAYER_MAX_SPEED)
        # Update player position
        self.player_y += self.player_speed

        # Prevent player from going past the walls
        self.player_y = jnp.clip(self.player_y, WALL_TOP_Y + WALL_TOP_HEIGHT - 8, WALL_BOTTOM_Y - 4)
        self.player_y = jnp.round(self.player_y)


    def ball_step(self):
        """
        Updates the ball's position and handles bouncing on walls and paddles.
        """
        # Update ball position
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Bounce on top and bottom walls
        if self.ball_y <= WALL_TOP_Y + WALL_TOP_HEIGHT or self.ball_y >= WALL_BOTTOM_Y - BALL_SIZE[1]:
            self.ball_vel_y = -self.ball_vel_y

        # Bounce on player paddle
        if PLAYER_X <= self.ball_x <= PLAYER_X + PLAYER_SIZE[0] and self.ball_vel_x > 0:
            if self.player_y - BALL_SIZE[1] <= self.ball_y <= self.player_y + PLAYER_SIZE[1] + BALL_SIZE[1]:
                self.ball_vel_x = -self.ball_vel_x

        # Bounce on enemy paddle
        if ENEMY_X <= self.ball_x <= ENEMY_X + ENEMY_SIZE[0]  and self.ball_vel_x < 0:
            if self.enemy_y - BALL_SIZE[1] <= self.ball_y <= self.enemy_y + ENEMY_SIZE[1] + BALL_SIZE[1]:
                self.ball_vel_x = -self.ball_vel_x

        # Ball goes past player paddle
        if self.ball_x > PLAYER_X + PLAYER_SIZE[0]:
            self.enemy_score += 1
            self.reset_ball_position()

        # Ball goes past enemy paddle
        if self.ball_x < ENEMY_X - ENEMY_SIZE[0]:
            self.player_score += 1
            self.reset_ball_position()

    def reset_ball_position(self):
        """
        Resets the ball position to the center of the game area.
        """
        self.ball_x, self.ball_y = 78, 115  # Reset ball position
        self.ball_vel_x, self.ball_vel_y = BALL_SPEED[0], BALL_SPEED[1]  # Reset ball velocity

    def enemy_step(self):
        """
        Updates the enemy paddle's y-coordinate to track the ball's position.
        """
        if self.step_counter % 8:
            direction = jnp.sign(self.ball_y - self.enemy_y)

            if direction < 0:
                self.enemy_speed -= ENEMY_ACCELERATION
            elif direction > 0:
                self.enemy_speed += ENEMY_ACCELERATION
            else:
                self.enemy_speed *= 0.9  # Apply friction when no action is taken

            # Limit player speed to the maximum allowed value
            self.enemy_speed = jnp.clip(self.enemy_speed, -ENEMY_MAX_SPEED, ENEMY_MAX_SPEED)
            # Update player position
            self.enemy_y += self.enemy_speed

            self.enemy_y = jnp.clip(self.enemy_y, WALL_TOP_Y + WALL_TOP_HEIGHT - 8, WALL_BOTTOM_Y - 4)

    def step(self, action):
        """
        Performs a single step of the game.
        """
        self.step_counter += 1
        self.player_step(action)
        self.ball_step()
        self.enemy_step()

    def jax_rendering(self):
        """
        Renders the current state of the game as a JAX array.
        """
        # Create a blank canvas with background color
        canvas = np.full((210, 160, 3), BACKGROUND_COLOR, dtype=np.uint8)

        # Draw player, ball, and enemy on the canvas
        if 0 <= int(self.player_y) < canvas.shape[0] - PLAYER_SIZE[1]:
            canvas[int(self.player_y):int(self.player_y) + PLAYER_SIZE[1], PLAYER_X:PLAYER_X + PLAYER_SIZE[0]] = PLAYER_COLOR  # Player paddle
        if 0 <= int(self.enemy_y) < canvas.shape[0] - ENEMY_SIZE[1]:
            canvas[int(self.enemy_y):int(self.enemy_y) + ENEMY_SIZE[1], ENEMY_X:ENEMY_X + ENEMY_SIZE[0]] = ENEMY_COLOR  # Enemy paddle
        if 0 <= int(self.ball_y) < canvas.shape[0] - BALL_SIZE[1] and 0 <= int(self.ball_x) < canvas.shape[1] - BALL_SIZE[0]:
            canvas[int(self.ball_y):int(self.ball_y) + BALL_SIZE[1], int(self.ball_x):int(self.ball_x) + BALL_SIZE[0]] = BALL_COLOR  # Ball

        # Draw walls
        canvas[WALL_TOP_Y:WALL_TOP_Y + WALL_TOP_HEIGHT, :] = WALL_COLOR  # Top wall
        canvas[WALL_BOTTOM_Y:WALL_BOTTOM_Y + WALL_BOTTOM_HEIGHT, :] = WALL_COLOR  # Bottom wall

        # Draw scores
        self.draw_score(canvas, self.player_score, position=(120, 2), color=PLAYER_COLOR)
        self.draw_score(canvas, self.enemy_score, position=(30, 2), color=ENEMY_COLOR)

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

    def display(self, screen):
        """
        Displays the rendered game state using pygame.
        """
        canvas = self.jax_rendering()
        canvas_np = np.array(canvas)
        canvas_np = np.flipud(np.rot90(canvas_np, k=1))  # Rotate 90 degrees counterclockwise and flip vertically to fix orientation
        pygame_surface = pygame.surfarray.make_surface(canvas_np)
        screen.blit(pygame.transform.scale(pygame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))
        pygame.display.flip()

    def get_state(self):
        """
        Returns the current state of the game.
        """
        return [self.player_y, self.ball_x, self.ball_y, self.enemy_y, self.ball_vel_x, self.ball_vel_y, self.player_score, self.enemy_score]

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pong Game")
clock = pygame.time.Clock()

# Create a Game instance
game = Game()

# Run the game until the user quits
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = get_human_action()
    game.step(action)
    game.display(screen)
    clock.tick(60)  # Set frame rate to 60 FPS

# Quit Pygame
pygame.quit()
