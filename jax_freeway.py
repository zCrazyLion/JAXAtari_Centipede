from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List

# Actions
NOOP = 0
UP = 1
DOWN = 2


@dataclass
class GameConfig:
    """Game configuration parameters"""
    screen_width: int = 160
    screen_height: int = 210
    chicken_width: int = 6
    chicken_height: int = 8
    chicken_x: int = 40  # Fixed x position
    car_width: int = 8
    car_height: int = 10
    num_lanes: int = 10
    lane_spacing: int = 16
    car_speeds: List[float] = None

    def __post_init__(self):
        if self.car_speeds is None:
            # Upper 5 lanes move left (-), lower 5 lanes move right (+)
            # Each lane has a different speed
            self.car_speeds = [
                -3.0,   # Lane 0: Fast left
                -1.5,   # Lane 1: Slow left
                -2.2,   # Lane 2: Medium-fast left
                -1.8,   # Lane 3: Medium-slow left
                -2.5,   # Lane 4: Medium left
                1.2,    # Lane 5: Slow right
                2.8,    # Lane 6: Fast right
                1.8,    # Lane 7: Medium right
                2.2,    # Lane 8: Medium-fast right
                1.5     # Lane 9: Medium-slow right
            ]


class GameState(NamedTuple):
    """Represents the current state of the game"""
    chicken_y: chex.Array
    cars: chex.Array  # Shape: (num_lanes, 2) for x,y positions
    score: chex.Array
    time: chex.Array
    cooldown: chex.Array  # Cooldown after collision
    game_over: chex.Array


class FreewayGameLogic:
    def __init__(self, config: GameConfig):
        self.config = config
        self.state = self.reset()

    def reset(self) -> GameState:
        """Initialize a new game state"""
        # Start chicken at bottom
        chicken_y = self.config.screen_height - self.config.chicken_height - 20

        # Initialize one car per lane
        cars = []
        for lane in range(self.config.num_lanes):
            lane_y = self.config.lane_spacing * (lane + 1)
            # Upper 5 lanes start from right, lower 5 lanes start from left
            if lane < 5:
                x = self.config.screen_width  # Start from right
            else:
                x = -self.config.car_width  # Start from left
            cars.append([x, lane_y])

        return GameState(
            chicken_y=jnp.array(chicken_y),
            cars=jnp.array(cars),
            score=jnp.array(0),
            time=jnp.array(0.0),
            cooldown=jnp.array(0),
            game_over=jnp.array(False)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> GameState:
        """Take a step in the game given an action"""
        # Update chicken position if not in cooldown
        dy = jnp.where(state.cooldown > 0, 0.0,
                       jnp.where(action == UP, -2.0,
                                 jnp.where(action == DOWN, 2.0, 0.0)))

        new_y = jnp.clip(state.chicken_y + dy,
                         0,
                         self.config.screen_height - self.config.chicken_height)

        # Update car positions
        new_cars = state.cars
        for lane in range(self.config.num_lanes):
            # Update x position based on lane speed
            new_x = state.cars[lane, 0] + self.config.car_speeds[lane]

            # Wrap around screen
            new_x = jnp.where(
                self.config.car_speeds[lane] > 0,
                jnp.where(new_x > self.config.screen_width,
                          -self.config.car_width,
                          new_x),
                jnp.where(new_x < -self.config.car_width,
                          self.config.screen_width,
                          new_x)
            )

            new_cars = new_cars.at[lane, 0].set(new_x)

        # Check for collisions
        def check_collision(car_pos):
            car_x, car_y = car_pos
            dx = jnp.abs(self.config.chicken_x - car_x)
            dy = jnp.abs(new_y - car_y)
            return jnp.logical_and(
                dx < (self.config.chicken_width + self.config.car_width) / 2,
                dy < (self.config.chicken_height + self.config.car_height) / 2
            )

        # Check collisions for all cars
        collisions = jax.vmap(check_collision)(new_cars)
        any_collision = jnp.any(collisions)

        # Update cooldown
        new_cooldown = jnp.where(any_collision,
                                 30,  # Set cooldown frames after collision
                                 jnp.maximum(0, state.cooldown - 1))

        # Update score if chicken reaches top
        new_score = jnp.where(new_y <= 0,
                              state.score + 1,
                              state.score)

        # Reset chicken position if scored
        new_y = jnp.where(new_y <= 0,
                          self.config.screen_height - self.config.chicken_height,
                          new_y)

        # Update time
        new_time = state.time + 1.0 / 60

        # Check game over (optional: could be based on time or score limit)
        game_over = jnp.where(new_time >= 120.0,  # 2 minute time limit
                              jnp.array(True),
                              state.game_over)

        return GameState(
            chicken_y=new_y,
            cars=new_cars,
            score=new_score,
            time=new_time,
            cooldown=new_cooldown,
            game_over=game_over
        )


@dataclass
class RenderConfig:
    """Configuration for rendering"""
    scale_factor: int = 4
    background_color: Tuple[int, int, int] = (0, 0, 0)
    chicken_color: Tuple[int, int, int] = (255, 255, 255)
    car_colors: List[Tuple[int, int, int]] = None
    text_color: Tuple[int, int, int] = (255, 255, 255)
    road_color: Tuple[int, int, int] = (50, 50, 50)
    line_color: Tuple[int, int, int] = (255, 255, 0)

    def __post_init__(self):
        if self.car_colors is None:
            # Different colors for upper and lower lanes
            self.car_colors = (
                    [(255, 0, 0)] * 5 +  # Upper lanes (red)
                    [(0, 255, 0)] * 5  # Lower lanes (green)
            )


class GameRenderer:
    def __init__(self, game_config: GameConfig, render_config: RenderConfig):
        self.game_config = game_config
        self.render_config = render_config

        pygame.init()
        self.screen = pygame.display.set_mode((
            self.game_config.screen_width * self.render_config.scale_factor,
            self.game_config.screen_height * self.render_config.scale_factor
        ))
        pygame.display.set_caption("JAX Freeway")

        self.chicken_sprite = self._create_chicken_sprite()
        self.car_sprites = self._create_car_sprites()
        self.font = pygame.font.Font(None, 36)

    def _create_chicken_sprite(self) -> pygame.Surface:
        size = (self.game_config.chicken_width * self.render_config.scale_factor,
                self.game_config.chicken_height * self.render_config.scale_factor)
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        # Simple triangle for the chicken
        pygame.draw.polygon(sprite, self.render_config.chicken_color, [
            (size[0] // 2, 0),
            (0, size[1]),
            (size[0], size[1])
        ])

        return sprite

    def _create_car_sprites(self) -> List[pygame.Surface]:
        sprites = []
        size = (self.game_config.car_width * self.render_config.scale_factor,
                self.game_config.car_height * self.render_config.scale_factor)

        for color in self.render_config.car_colors:
            sprite = pygame.Surface(size, pygame.SRCALPHA)

            # Draw car body
            pygame.draw.rect(sprite, color,
                             (0, size[1] // 4, size[0], size[1] // 2))
            # Draw wheels
            wheel_radius = size[1] // 6
            pygame.draw.circle(sprite, (0, 0, 0),
                               (wheel_radius * 2, size[1] * 3 // 4), wheel_radius)
            pygame.draw.circle(sprite, (0, 0, 0),
                               (size[0] - wheel_radius * 2, size[1] * 3 // 4),
                               wheel_radius)

            sprites.append(sprite)

        return sprites

    def render(self, state: GameState):
        """Render the current game state"""
        self.screen.fill(self.render_config.background_color)

        # Draw road
        road_rect = pygame.Rect(
            0,
            self.game_config.lane_spacing * self.render_config.scale_factor,
            self.game_config.screen_width * self.render_config.scale_factor,
            (self.game_config.num_lanes * self.game_config.lane_spacing) *
            self.render_config.scale_factor
        )
        pygame.draw.rect(self.screen, self.render_config.road_color, road_rect)

        # Draw lane lines
        for lane in range(self.game_config.num_lanes + 1):
            y = (self.game_config.lane_spacing * lane + self.game_config.lane_spacing) * \
                self.render_config.scale_factor
            pygame.draw.line(
                self.screen,
                self.render_config.line_color,
                (0, y),
                (self.game_config.screen_width * self.render_config.scale_factor, y),
                2
            )

        # Draw cars
        for lane in range(self.game_config.num_lanes):
            car_sprite = self.car_sprites[lane]
            if self.game_config.car_speeds[lane] < 0:  # Flip sprite if moving left
                car_sprite = pygame.transform.flip(car_sprite, True, False)

            car_pos = (
                int(state.cars[lane][0] * self.render_config.scale_factor),
                int(state.cars[lane][1] * self.render_config.scale_factor -
                    self.game_config.car_height * self.render_config.scale_factor // 2)
            )
            self.screen.blit(car_sprite, car_pos)

        # Draw chicken
        chicken_pos = (
            int(self.game_config.chicken_x * self.render_config.scale_factor -
                self.game_config.chicken_width * self.render_config.scale_factor // 2),
            int(state.chicken_y * self.render_config.scale_factor)
        )
        self.screen.blit(self.chicken_sprite, chicken_pos)

        # Draw UI
        score_text = self.font.render(f"Score: {state.score}",
                                      True,
                                      self.render_config.text_color)
        time_text = self.font.render(f"Time: {state.time:.1f}",
                                     True,
                                     self.render_config.text_color)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 40))

        if state.game_over:
            game_over_text = self.font.render(
                "Game Over!",
                True,
                self.render_config.text_color
            )
            text_rect = game_over_text.get_rect(center=(
                self.game_config.screen_width * self.render_config.scale_factor // 2,
                self.game_config.screen_height * self.render_config.scale_factor // 2
            ))
            self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()

    def close(self):
        """Clean up pygame resources"""
        pygame.quit()


def main():
    # Create configurations
    game_config = GameConfig()
    render_config = RenderConfig()

    # Initialize game and renderer
    game = FreewayGameLogic(game_config)
    renderer = GameRenderer(game_config, render_config)

    # Setup game loop
    clock = pygame.time.Clock()
    running = True

    while running and not game.state.game_over:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = UP
        elif keys[pygame.K_s]:
            action = DOWN
        else:
            action = NOOP

        # Update game state
        game.state = game.step(game.state, action)

        # Render
        renderer.render(game.state)

        # Cap at 60 FPS
        clock.tick(60)

    # If game over, wait before closing
    if game.state.game_over:
        pygame.time.wait(2000)

    renderer.close()


if __name__ == "__main__":
    main()