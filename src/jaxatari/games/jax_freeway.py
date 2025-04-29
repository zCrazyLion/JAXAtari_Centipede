from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple, List

from jaxatari.environment import JaxEnvironment

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
    lane_borders: List[int] = None
    top_border: int = 15
    top_path: int = 8
    bottom_border: int = 180

    def __post_init__(self):
        if self.car_speeds is None:
            # Upper 5 lanes move left (-), lower 5 lanes move right (+)
            # Value at i is the frequency in which car at lane i moves one pixel
            self.car_update = [
                -5,  # Lane 0
                -4,  # Lane 1
                -3,  # Lane 2
                -2,  # Lane 3
                -1,  # Lane 4
                1,  # Lane 5
                2,  # Lane 6
                3,  # Lane 7
                4,  # Lane 8
                5,  # Lane 9
            ]
        if self.lane_borders is None:
            # Upper 5 lanes move left (-), lower 5 lanes move right (+)
            # Value at i is the frequency in which car at lane i moves one pixel
            self.lane_borders = [
                self.top_border + self.top_path,  # Lane 0
                1 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 1
                2 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 2
                3 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 3
                4 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 4
                5 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 5
                6 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 6
                7 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 7
                8 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 8
                9 * self.lane_spacing + (self.top_border + self.top_path),  # Lane 10
                10 * self.lane_spacing
                + (self.top_border + self.top_path)
                + 2,  # Lane 10
            ]


class GameState(NamedTuple):
    """Represents the current state of the game"""

    chicken_y: chex.Array
    cars: chex.Array  # Shape: (num_lanes, 2) for x,y positions
    score: chex.Array
    time: chex.Array
    cooldown: chex.Array  # Cooldown after collision
    game_over: chex.Array


class EntityPosition(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    width: jnp.ndarray
    height: jnp.ndarray


class FreewayObservation(NamedTuple):
    chicken: EntityPosition
    car: EntityPosition
    score: jnp.ndarray


class FreewayInfo(NamedTuple):
    time: jnp.ndarray


class FreewayGameLogic(JaxEnvironment[GameState, FreewayObservation, FreewayInfo]):
    def __init__(self):
        super().__init__()
        self.config = GameConfig()
        self.state = self.reset()

    def reset(self) -> Tuple[GameState, FreewayObservation]:
        """Initialize a new game state"""
        # Start chicken at bottom
        chicken_y = self.config.bottom_border + self.config.chicken_height - 1
        # Initialize one car per lane
        cars = []
        for lane in range(self.config.num_lanes):
            lane_y = (
                self.config.lane_borders[lane]
                + int(self.config.lane_spacing / 2)
                - int(self.config.car_height / 2)
            )
            # Upper 5 lanes start from right, lower 5 lanes start from left
            if lane < 5:
                x = (
                    self.config.screen_width - self.config.car_width + 0
                )  # Start from right
            else:
                x = 0  # Start from left
            cars.append([x, lane_y])

        state = GameState(
            chicken_y=jnp.array(chicken_y),
            cars=jnp.array(cars),
            score=jnp.array(0),
            time=jnp.array(0),
            cooldown=jnp.array(0),
            game_over=jnp.array(False),
        )

        return state, self._get_observation(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> GameState:
        """Take a step in the game given an action"""
        # Update chicken position if not in cooldown
        dy = jnp.where(
            state.cooldown > 0,
            1.0,
            jnp.where(action == UP, -1.0, jnp.where(action == DOWN, 1.0, 0.0)),
        )

        new_y = jnp.clip(
            state.chicken_y + dy,
            self.config.top_border,
            self.config.bottom_border + self.config.chicken_height - 1,
        )

        # Update car positions
        new_cars = state.cars
        for lane in range(self.config.num_lanes):
            # Update x position based on lane speed
            dir = (
                self.config.car_update[lane] / jnp.abs(self.config.car_update[lane])
            ).astype(jnp.int32)
            new_x = jax.lax.cond(
                jnp.equal(jnp.mod(state.time, self.config.car_update[lane]), 0),
                lambda: state.cars[lane, 0] + dir,
                lambda: state.cars[lane, 0],
            )

            # Wrap around screen
            new_x = jnp.where(
                self.config.car_update[lane] > 0,
                jnp.where(
                    new_x > self.config.screen_width, -self.config.car_width, new_x
                ),
                jnp.where(
                    new_x < -self.config.car_width, self.config.screen_width, new_x
                ),
            )

            new_cars = new_cars.at[lane, 0].set(new_x)

        # Check for collisions
        def check_collision(car_pos):
            car_x, car_y = car_pos
            return jnp.logical_and(
                self.config.chicken_x < car_x + self.config.car_width,
                jnp.logical_and(
                    self.config.chicken_x + self.config.chicken_width > car_x,
                    jnp.logical_and(
                        state.chicken_y - self.config.chicken_height < car_y,
                        state.chicken_y > car_y - self.config.car_height,
                    ),
                ),
            )

        # Check collisions for all cars
        collisions = jax.vmap(check_collision)(new_cars)
        any_collision = jnp.any(collisions)
        any_collision = jax.lax.cond(
            state.cooldown > 0, lambda _: False, lambda _: any_collision, operand=None
        )

        # Update cooldown
        new_cooldown = jnp.where(
            any_collision,
            24,  # Set cooldown frames after collision
            jnp.maximum(0, state.cooldown - 1),
        )

        # Update score if chicken reaches top
        new_score = jnp.where(
            new_y <= self.config.top_border, state.score + 1, state.score
        )

        # Reset chicken position if scored
        new_y = jnp.where(
            new_y <= self.config.top_border,
            self.config.bottom_border + self.config.chicken_height - 1,
            new_y,
        )

        # Update time
        new_time = state.time + 1

        # Check game over (optional: could be based on time or score limit)
        game_over = jnp.where(
            new_time >= 255 * 32,  # 2 minute time limit
            jnp.array(True),
            state.game_over,
        )

        new_state = GameState(
            chicken_y=new_y,
            cars=new_cars,
            score=new_score,
            time=new_time,
            cooldown=new_cooldown,
            game_over=game_over,
        )
        done = self._get_done(new_state)
        reward = self._get_reward(state, new_state)
        obs = self._get_observation(new_state)
        info = self._get_info(new_state)

        return new_state, obs, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: GameState):
        # create chicken
        chicken = EntityPosition(
            x=jnp.array(self.config.chicken_x),
            y=state.chicken_y,
            width=jnp.array(self.config.chicken_width),
            height=jnp.array(self.config.chicken_height),
        )

        # create cars
        cars = jnp.zeros((self.config.num_lanes, 4))
        for i in range(self.config.num_lanes):
            car_pos = state.cars.at[i].get()
            cars = cars.at[i].set(
                jnp.array(
                    [
                        car_pos.at[0].get(),  # x position
                        car_pos.at[1].get(),  # y position
                        self.config.car_width,  # width
                        self.config.car_height,  # height
                    ]
                )
            )
        return FreewayObservation(chicken=chicken, car=cars, score=state.score)

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: GameState) -> FreewayInfo:
        return FreewayInfo(time=state.time)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, previous_state: GameState, state: GameState):
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: GameState) -> bool:
        return state.game_over


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
            self.car_colors = [(255, 0, 0)] * 5 + [  # Upper lanes (red)
                (0, 255, 0)
            ] * 5  # Lower lanes (green)


class GameRenderer:
    def __init__(self, game_config: GameConfig, render_config: RenderConfig):
        self.game_config = game_config
        self.render_config = render_config

        pygame.init()
        self.screen = pygame.display.set_mode(
            (
                self.game_config.screen_width * self.render_config.scale_factor,
                self.game_config.screen_height * self.render_config.scale_factor,
            )
        )
        pygame.display.set_caption("JAX Freeway")

        self.chicken_sprite = self._create_chicken_sprite()
        self.car_sprites = self._create_car_sprites()
        self.font = pygame.font.Font(None, 36)

    def _create_chicken_sprite(self) -> pygame.Surface:
        size = (
            self.game_config.chicken_width * self.render_config.scale_factor,
            self.game_config.chicken_height * self.render_config.scale_factor,
        )
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        # Simple triangle for the chicken
        pygame.draw.rect(
            sprite, self.render_config.chicken_color, (0, 0, size[0], size[1])
        )

        return sprite

    def _create_car_sprites(self) -> List[pygame.Surface]:
        sprites = []
        size = (
            self.game_config.car_width * self.render_config.scale_factor,
            self.game_config.car_height * self.render_config.scale_factor,
        )

        for color in self.render_config.car_colors:
            sprite = pygame.Surface(size, pygame.SRCALPHA)

            # Draw car body
            pygame.draw.rect(sprite, color, (0, 0, size[0], size[1]))

            sprites.append(sprite)

        return sprites

    def render(self, state: GameState):
        """Render the current game state"""
        self.screen.fill(self.render_config.background_color)

        # Draw road
        road_rect = pygame.Rect(
            0,
            self.game_config.top_border * self.render_config.scale_factor,
            self.game_config.screen_width * self.render_config.scale_factor,
            (
                self.game_config.screen_height
                - (self.game_config.screen_height - self.game_config.bottom_border)
            )
            * self.render_config.scale_factor,
        )
        pygame.draw.rect(self.screen, self.render_config.road_color, road_rect)

        # Draw lane lines
        for lane in range(self.game_config.num_lanes + 1):
            y = (self.game_config.lane_borders[lane]) * self.render_config.scale_factor

            pygame.draw.line(
                self.screen,
                self.render_config.line_color,
                (0, y),
                (self.game_config.screen_width * self.render_config.scale_factor, y),
                3,
            )

        # Draw cars
        for lane in range(self.game_config.num_lanes):
            car_sprite = self.car_sprites[lane]

            car_pos = (
                int(state.cars[lane][0] * self.render_config.scale_factor),
                int(state.cars[lane][1] * self.render_config.scale_factor),
            )
            self.screen.blit(car_sprite, car_pos)

        # Draw chicken
        chicken_rect = pygame.Rect(
            int(self.game_config.chicken_x * self.render_config.scale_factor),
            int(state.chicken_y * self.render_config.scale_factor),
            int(self.game_config.chicken_width * self.render_config.scale_factor),
            int(self.game_config.chicken_height * self.render_config.scale_factor),
        )

        pygame.draw.rect(self.screen, self.render_config.chicken_color, chicken_rect)

        # Draw UI
        score_text = self.font.render(
            f"Score: {state.score}", True, self.render_config.text_color
        )
        self.screen.blit(score_text, (10, 10))

        if state.game_over:
            game_over_text = self.font.render(
                "Game Over!", True, self.render_config.text_color
            )
            text_rect = game_over_text.get_rect(
                center=(
                    self.game_config.screen_width
                    * self.render_config.scale_factor
                    // 2,
                    self.game_config.screen_height
                    * self.render_config.scale_factor
                    // 2,
                )
            )
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
    game = FreewayGameLogic()
    state, init_obs = game.reset()
    renderer = GameRenderer(game_config, render_config)

    # Setup game loop
    clock = pygame.time.Clock()
    running = True
    done = False

    while running and not done:
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
        state, obs, reward, done, info = game.step(state, action)

        # Render
        renderer.render(state)

        # Cap at 60 FPS
        clock.tick(60)

    # If game over, wait before closing
    if done:
        pygame.time.wait(2000)

    renderer.close()


if __name__ == "__main__":
    main()
