import sys

import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from BaseGame import BaseGame
import pygame


class PongGame(BaseGame):
    # State array indices
    BALL_X = 0
    BALL_Y = 1
    BALL_VX = 2
    BALL_VY = 3
    PADDLE_CPU_Y = 4
    PADDLE_PLAYER_Y = 5
    SCORE_CPU = 6
    SCORE_PLAYER = 7
    TIME = 8
    STATE_SIZE = 9

    def __init__(self, width=160, height=210):
        BaseGame.__init__(self)
        self.width = width
        self.height = height
        self.paddle_height = 25
        self.paddle_width = 4
        self.ball_size = 4
        self.paddle_speed = 2.0
        self.ball_speed = 2

        # Initialize game state array
        self.initial_state = jnp.array(
            [
                width / 2,  # ball_x
                height / 2,  # ball_y
                self.ball_speed,  # ball_vx
                0.0,  # ball_vy
                height / 2,  # cpu_paddle_y
                height / 2,  # player_paddle_y
                0,  # cpu_score
                0,  # player_score
                0,  # time
            ]
        )

        self.curr_state = self.initial_state

    def get_oc_state(self):
        return jnp.array(
            [
                self.curr_state[PongGame.PADDLE_PLAYER_Y],
                self.curr_state[PongGame.PADDLE_CPU_Y],
                self.curr_state[PongGame.BALL_X], self.curr_state[PongGame.BALL_Y],
                self.curr_state[PongGame.SCORE_PLAYER], self.curr_state[PongGame.SCORE_CPU],
            ]
        )

    @staticmethod
    @jax.jit
    def update_paddle(
        pos: float, movement: float, height: float, paddle_height: float
    ) -> float:
        """Update paddle position with bounds checking"""
        new_pos = pos + movement
        return jnp.clip(new_pos, paddle_height / 2, height - paddle_height / 2)

    @staticmethod
    @jax.jit
    def update_cpu(
        curr_pos, ball_pos, height: float, paddle_height: float, time: float
    ) -> float:
        new_pos = jnp.where(
            jnp.less(curr_pos - (paddle_height / 2), ball_pos[1]),
            curr_pos,
            curr_pos - 1,
        )
        new_pos = jnp.where(
            jnp.greater(curr_pos + (paddle_height / 2), ball_pos[1]),
            new_pos,
            new_pos + 1,
        )
        new_pos = jnp.where((time % 30 == 0), curr_pos, new_pos)

        return jnp.clip(new_pos, paddle_height / 2, height - paddle_height / 2)

    @staticmethod
    @jax.jit
    def check_paddle_collision(
        ball_pos: jnp.ndarray,
        ball_vel: jnp.ndarray,
        paddle_pos: float,
        is_left_paddle: bool,
        paddle_x: float,
        paddle_height: float,
        paddle_width: float,
        ball_size: float,
    ) -> Tuple[bool, jnp.ndarray]:
        """Check and resolve ball collision with paddle"""
        paddle_y_min = paddle_pos - paddle_height / 2
        paddle_y_max = paddle_pos + paddle_height / 2

        # Replace if/else with jnp.where for left/right paddle collision check
        left_paddle_check = ball_pos[0] - ball_size / 2 <= paddle_x + paddle_width
        right_paddle_check = ball_pos[0] + ball_size / 2 >= paddle_x
        x_collision = jnp.where(is_left_paddle, left_paddle_check, right_paddle_check)

        collision = jnp.logical_and(
            x_collision,
            jnp.logical_and(ball_pos[1] >= paddle_y_min, ball_pos[1] <= paddle_y_max),
        )

        # On collision, reverse x velocity and add some y velocity based on hit position
        relative_intersect_y = (paddle_pos - ball_pos[1]) / (paddle_height / 2)
        bounce_angle = relative_intersect_y * jnp.pi / 4

        speed = jnp.sqrt(ball_vel[0] ** 2 + ball_vel[1] ** 2)
        new_vel = jnp.where(
            collision,
            jnp.array([-ball_vel[0], -jnp.sin(bounce_angle) * speed]),
            ball_vel,
        )

        return collision, new_vel

    @staticmethod
    @jax.jit
    def update_state(
        state: jnp.ndarray,
        player_movement: float,
        game_params: Dict,
        spacebar_released: bool = False,
    ) -> jnp.ndarray:
        """Update game state for one frame"""
        width = game_params["width"]
        height = game_params["height"]
        paddle_height = game_params["paddle_height"]
        paddle_width = game_params["paddle_width"]
        ball_size = game_params["ball_size"]
        ball_speed = game_params["ball_speed"]

        # Extract current state
        ball_pos = state[PongGame.BALL_X : PongGame.BALL_Y + 1]
        ball_vel = state[PongGame.BALL_VX : PongGame.BALL_VY + 1]
        cpu_paddle_pos = state[PongGame.PADDLE_CPU_Y]
        player_paddle_pos = state[PongGame.PADDLE_PLAYER_Y]
        score = state[PongGame.SCORE_CPU : PongGame.SCORE_PLAYER + 1]
        time = state[PongGame.TIME]

        # Update paddle positions
        new_player_paddle_pos = PongGame.update_paddle(
            player_paddle_pos, player_movement, height, paddle_height
        )

        # Update ball position
        new_ball_pos = ball_pos + ball_vel

        new_cpu_paddle_pos = PongGame.update_cpu(
            cpu_paddle_pos, ball_pos, height, paddle_height, time
        )

        # Check paddle collisions
        cpu_paddle_collision, new_vel_after_cpu_paddle = (
            PongGame.check_paddle_collision(
                new_ball_pos,
                ball_vel,
                new_cpu_paddle_pos,
                True,
                paddle_width,
                paddle_height,
                paddle_width,
                ball_size,
            )
        )

        player_paddle_collision, new_vel_after_player_paddle = (
            PongGame.check_paddle_collision(
                new_ball_pos,
                new_vel_after_cpu_paddle,
                new_player_paddle_pos,
                False,
                width - paddle_width,
                paddle_height,
                paddle_width,
                ball_size,
            )
        )

        # Combine paddle collision velocities
        new_vel = jnp.where(
            cpu_paddle_collision | player_paddle_collision,
            jnp.where(
                cpu_paddle_collision,
                new_vel_after_cpu_paddle,
                new_vel_after_player_paddle,
            ),
            ball_vel,
        )

        # Wall collisions
        wall_collision = jnp.logical_or(
            new_ball_pos[1] <= ball_size / 2, new_ball_pos[1] >= height - ball_size / 2
        )
        new_vel = new_vel.at[1].multiply(jnp.where(wall_collision, -1.0, 1.0))

        # Check if ball is moving away from the paddle (should be moving left from right paddle)
        moving_away = jnp.less(
            ball_vel[0], 0.0
        )  # Moving left after hitting right paddle

        # Define "near paddle" zone for right paddle
        near_player_paddle = jnp.less(
            jnp.abs(new_ball_pos[0] - (width - paddle_width)), ball_size * 2
        )

        # Check for power shot conditions for right paddle only
        power_shot = (
            spacebar_released
            & (player_paddle_collision | near_player_paddle)
            & moving_away
        )

        # Apply power shot
        new_vel = jnp.where(power_shot, new_vel * 1.25, new_vel)

        # Score update and ball reset conditions
        scored_cpu = new_ball_pos[0] >= width - ball_size / 2
        scored_player = new_ball_pos[0] <= ball_size / 2

        new_score = score + jnp.array([scored_cpu, scored_player])

        # Reset ball if scored
        reset_ball = jnp.logical_or(scored_cpu, scored_player)
        new_ball_pos = jnp.where(
            reset_ball, jnp.array([width / 2, height / 2]), new_ball_pos
        )
        new_vel = jnp.where(
            reset_ball,
            jnp.array([ball_speed * jnp.where(scored_player, 1.0, -1.0), 0.0]),
            new_vel,
        )

        # Construct new state array
        new_state = jnp.concatenate(
            [
                new_ball_pos,
                new_vel,
                jnp.array([new_cpu_paddle_pos, new_player_paddle_pos]),
                new_score,
                jnp.array([time + 1]),
            ]
        )

        return new_state

    def get_game_params(self) -> Dict:
        """Return dictionary of game parameters"""
        return {
            "width": self.width,
            "height": self.height,
            "paddle_height": self.paddle_height,
            "paddle_width": self.paddle_width,
            "ball_size": self.ball_size,
            "ball_speed": self.ball_speed,
        }

    def reset(self) -> jnp.ndarray:
        """Reset game to initial state"""
        return self.initial_state

    def step(
        self, state: jnp.ndarray, p1_movement: float, spacebar_released: bool
    ) -> jnp.ndarray:
        """Take one game step"""
        self.curr_state = state
        return self.update_state(
            state, p1_movement, self.get_game_params(), spacebar_released
        )


class PongRenderer:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("JAX Pong")
        self.clock = pygame.time.Clock()

    def render(self, state: jnp.ndarray, game_params: Dict):
        """Render the current game state"""
        self.screen.fill(self.BLACK)

        # Draw paddles
        paddle_width = game_params["paddle_width"]
        paddle_height = game_params["paddle_height"]

        # Left paddle (CPU)
        paddle1_y = float(state[PongGame.PADDLE_CPU_Y])
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(0, paddle1_y - paddle_height / 2, paddle_width, paddle_height),
        )

        # Right paddle (Player)
        paddle2_y = float(state[PongGame.PADDLE_PLAYER_Y])
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            pygame.Rect(
                self.width - paddle_width,
                paddle2_y - paddle_height / 2,
                paddle_width,
                paddle_height,
            ),
        )

        # Draw ball
        ball_size = game_params["ball_size"]
        ball_x = float(state[PongGame.BALL_X])
        ball_y = float(state[PongGame.BALL_Y])
        pygame.draw.circle(self.screen, self.WHITE, (ball_x, ball_y), ball_size / 2)

        # Draw center line
        for i in range(0, self.height, 20):
            pygame.draw.rect(
                self.screen, self.WHITE, pygame.Rect(self.width / 2 - 1, i, 2, 10)
            )

        # Draw scores
        font = pygame.font.Font(None, 74)
        score1 = int(state[PongGame.SCORE_CPU])
        score2 = int(state[PongGame.SCORE_PLAYER])

        score1_text = font.render(str(score1), True, self.WHITE)
        score2_text = font.render(str(score2), True, self.WHITE)

        self.screen.blit(score1_text, (self.width / 4, 20))
        self.screen.blit(score2_text, (3 * self.width / 4, 20))

        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS

    def close(self):
        pygame.quit()


def main():
    # Initialize game and renderer
    width, height = 160, 210
    game = PongGame(width, height)
    renderer = PongRenderer(width, height)
    game_params = game.get_game_params()
    state = game.reset()

    # Game loop
    try:
        frame_counter = 0
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Handle paddle movement
            keys = pygame.key.get_pressed()
            p1_movement = 0.0
            if keys[pygame.K_w]:
                p1_movement -= game.paddle_speed
            if keys[pygame.K_s]:
                p1_movement += game.paddle_speed

            p2_movement = 0.0
            if keys[pygame.K_UP]:
                p2_movement -= game.paddle_speed
            if keys[pygame.K_DOWN]:
                p2_movement += game.paddle_speed

            spacebar_released = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    spacebar_released = True
                    frame_counter = 3

            if frame_counter > 0:
                frame_counter -= 1
                spacebar_released = True

            # Update game state
            state = game.step(state, p1_movement, spacebar_released)

            # Render
            renderer.render(state, game_params)

    except KeyboardInterrupt:
        renderer.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
