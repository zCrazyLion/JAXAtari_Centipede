from functools import partial
import pygame
import chex
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple, NamedTuple
import random

NOOP = 0
LEFT = 1
RIGHT = 2

BOTTOM_BORDER = 176
TOP_BORDER = 23


@dataclass
class GameConfig:
    """Game configuration parameters"""
    screen_width: int = 160
    screen_height: int = 210
    skier_width: int = 10
    skier_height: int = 18
    skier_y:int = 40
    flag_width: int = 5
    flag_height: int = 14
    flag_distance: int = 20
    tree_width: int = 16
    tree_height: int = 30
    rock_width: int = 16
    rock_height: int = 7
    max_num_flags: int = 2
    max_num_trees: int = 4
    max_num_rocks: int = 3
    speed: float = 1.0


class GameState(NamedTuple):
    """Represents the current state of the game"""
    skier_x: chex.Array
    skier_pos: chex.Array # --> --_  \   |   |   /  _-- <--
    skier_fell: chex.Array
    flags: chex.Array
    trees: chex.Array
    rocks: chex.Array
    score: chex.Array
    time: chex.Array
    game_over: chex.Array


class SkiingGameLogic:
    def __init__(self, config: GameConfig):
        self.config = config
        self.state = self.reset()

    def reset(self) -> GameState:
        """Initialize a new game state"""
        flags = []
        y_spacing = (self.config.screen_height - 4 * self.config.flag_height) / self.config.max_num_flags
        for i in range(self.config.max_num_flags):
            x = random.uniform(self.config.flag_width, self.config.screen_width - self.config.flag_width - self.config.flag_distance)
            y = (i + 1) * y_spacing + self.config.flag_height
            flags.append((x, y))

        trees = []
        for _ in range(self.config.max_num_trees):
            x = random.uniform(self.config.tree_width, self.config.screen_width - self.config.tree_width)
            y = random.uniform(self.config.tree_height, self.config.screen_height - self.config.tree_height)
            trees.append((x, y))

        rocks = []
        for _ in range(self.config.max_num_rocks):
            x = random.uniform(self.config.rock_width, self.config.screen_width - self.config.rock_width)
            y = random.uniform(self.config.rock_height, self.config.screen_height - self.config.rock_height)
            rocks.append((x, y))

        return GameState(
            skier_x=jnp.array(76.0),
            skier_pos=jnp.array(4),
            skier_fell=jnp.array(0),
            flags=jnp.array(flags),
            trees=jnp.array(trees),
            rocks=jnp.array(rocks),
            score=jnp.array(20),
            time=jnp.array(0.0),
            game_over=jnp.array(False)
        )

    def _create_new_objs(self):
        flags = []
        trees = []
        rocks = []

        x = random.uniform(self.config.flag_width, self.config.screen_width - self.config.flag_width - self.config.flag_distance)
        flags.append((x, BOTTOM_BORDER))

        x = random.uniform(self.config.tree_width, self.config.screen_width - self.config.tree_width)
        trees.append((x, BOTTOM_BORDER))

        x = random.uniform(self.config.rock_width, self.config.screen_width - self.config.rock_width)
        rocks.append((x, BOTTOM_BORDER))

        return jnp.array(flags), jnp.array(trees), jnp.array(rocks)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: GameState, action: int) -> GameState:
        SPEED_DICT = jnp.array([
            0.5,
            0.8,
            0.9,
            1.0,
            1.0,
            0.9,
            0.8,
            0.5
        ], jnp.float32)


        """Take a step in the game given an action"""
        dx = jnp.where(action == LEFT, -self.config.speed,
                       jnp.where(action == RIGHT, self.config.speed, 0.0))

        skier_pos = jax.lax.cond(jnp.equal(action, LEFT), lambda _: state.skier_pos-1, lambda _: state.skier_pos, operand=None)
        skier_pos = jax.lax.cond(jnp.equal(action, RIGHT), lambda _: state.skier_pos + 1, lambda _: skier_pos,
                                 operand=None)
        skier_pos = jnp.clip(skier_pos, 0, 7)
        dy = self.config.speed * SPEED_DICT.at[skier_pos].get()

        new_x = jnp.clip(state.skier_x + dx,
                         self.config.skier_width / 2,
                         self.config.screen_width - self.config.skier_width / 2)

        new_trees = self.state.trees
        for i in range(len(self.state.trees)):
            new_trees = new_trees.at[i, 1].set(state.trees[i][1] - dy)

        new_rocks = self.state.rocks
        for i in range(len(self.state.rocks)):
            new_rocks = new_rocks.at[i, 1].set(state.rocks[i][1] - dy)

        new_flags = self.state.flags
        for i in range(len(self.state.flags)):
            new_flags = new_flags.at[i, 1].set(state.flags[i][1] - dy)


        def check_collision_flag(flag_pos):
            fx, fy = flag_pos
            dx_0 = new_x - fx
            dy_0 = jnp.abs(self.config.skier_y - fy)
            return (dx_0 > 0) &(dx_0 < self.config.flag_distance) & (dy_0 < 1)

        def check_collision(obj_pos, x_distance=3, y_distance=1):
            x, y = obj_pos
            dx = jnp.abs(new_x - x)
            dy = jnp.abs(self.config.skier_y - y)
            return jnp.logical_and(dx < x_distance, dy < y_distance)


        f, t, r = self._create_new_objs()
        new_flags = jnp.where(new_flags < TOP_BORDER, f, new_flags)
        new_trees = jnp.where(new_trees < TOP_BORDER, t, new_trees)
        new_rocks = jnp.where(new_rocks < TOP_BORDER, r, new_rocks)

        passed_flags = jax.vmap(check_collision_flag)(jnp.array(new_flags))
        collisions_tree = jax.vmap(check_collision)(jnp.array(new_trees))
        collisions_rocks = jax.vmap(check_collision)(jnp.array(new_rocks))

        num_colls = jnp.sum(collisions_tree) + jnp.sum(collisions_rocks)

        new_x, skier_fell, num_colls, new_flags, new_trees, new_rocks = jax.lax.cond(
            jnp.greater(state.skier_fell, 0),
            lambda _:(state.skier_x, state.skier_fell - 1, 0, state.flags, state.trees, state.rocks),
            lambda _:(new_x, state.skier_fell, num_colls, new_flags, new_trees, new_rocks),
            operand=None)

        skier_fell = jax.lax.cond(jnp.logical_and(jnp.greater(num_colls, 0), jnp.equal(skier_fell, 0)), lambda _: jnp.array(60), lambda _: skier_fell, operand=None)

        new_score = state.score - jnp.sum(passed_flags)
        game_over = jax.lax.cond(jnp.equal(new_score, 0), lambda _: jnp.array(True), lambda _: jnp.array(False), operand=None)
        new_time = state.time + 1.0 / 60


        return GameState(
            skier_x=new_x,
            skier_pos=jnp.array(skier_pos),
            skier_fell=skier_fell,
            flags=jnp.array(new_flags),
            trees=jnp.array(new_trees),
            rocks=jnp.array(new_rocks),
            score=new_score,
            time=new_time,
            game_over=game_over
        )


@dataclass
class RenderConfig:
    """Configuration for rendering"""
    scale_factor: int = 4
    background_color: Tuple[int, int, int] = (255, 255, 255)
    skier_color = [(0, 0, 255), (0, 0, 100), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (100, 0, 255)]
    flag_color: Tuple[int, int, int] = (255, 0, 0)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    tree_color: Tuple[int, int, int] = (0, 100, 0)
    rock_color: Tuple[int, int, int] = (128, 128, 128)
    game_over_color: Tuple[int, int, int] = (255, 0, 0)


class GameRenderer:
    def __init__(self, game_config: GameConfig, render_config: RenderConfig):
        self.game_config = game_config
        self.render_config = render_config

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((
            self.game_config.screen_width * self.render_config.scale_factor,
            self.game_config.screen_height * self.render_config.scale_factor
        ))
        pygame.display.set_caption("JAX Skiing Game")

        # Create sprites
        self.skier_sprite = self._create_skier_sprite()
        self.flag_sprite = self._create_flag_sprite()
        self.rock_sprite = self._create_rock_sprite()
        self.tree_sprite = self._create_tree_sprite()
        self.font = pygame.font.Font(None, 36)

    def _create_skier_sprite(self) -> list[pygame.Surface]:
        sprites = []
        for i in range(8):
            size = (self.game_config.skier_width * self.render_config.scale_factor,
                    self.game_config.skier_height * self.render_config.scale_factor)
            sprite = pygame.Surface(size, pygame.SRCALPHA)

            scaled_width = size[0]
            scaled_height = size[1]

            pygame.draw.polygon(sprite, self.render_config.skier_color[i], [
                (scaled_width // 2, 0),
                (0, scaled_height),
                (scaled_width, scaled_height)
            ])

            pygame.draw.line(sprite, self.render_config.text_color,
                             (0, scaled_height),
                             (scaled_width, scaled_height), 2)

            sprites.append(sprite)

        return sprites

    def _create_flag_sprite(self) -> pygame.Surface:
        size = (self.game_config.flag_width * self.render_config.scale_factor,
                self.game_config.flag_height * self.render_config.scale_factor)
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        scaled_width = size[0]
        scaled_height = size[1]

        pygame.draw.line(sprite, self.render_config.text_color,
                         (scaled_width // 2, 0),
                         (scaled_width // 2, scaled_height), 2)

        pygame.draw.polygon(sprite, self.render_config.flag_color, [
            (scaled_width // 2, scaled_height // 4),
            (scaled_width, scaled_height // 2),
            (scaled_width // 2, scaled_height // 4 * 3)
        ])

        return sprite

    def _create_tree_sprite(self) -> pygame.Surface:
        size = (self.game_config.tree_width * self.render_config.scale_factor,
                self.game_config.tree_height * self.render_config.scale_factor)
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        scaled_width = size[0]
        scaled_height = size[1]

        # Tree trunk
        trunk_width = scaled_width // 3
        trunk_height = scaled_height // 3
        trunk_rect = pygame.Rect(
            (scaled_width - trunk_width) // 2,
            scaled_height - trunk_height,
            trunk_width,
            trunk_height
        )
        pygame.draw.rect(sprite, (139, 69, 19), trunk_rect)

        # Tree triangles
        for i in range(3):
            height_offset = i * (scaled_height // 3)
            width_factor = 0.8 + (i * 0.1)
            pygame.draw.polygon(sprite, self.render_config.tree_color, [
                (scaled_width // 2, height_offset),
                (scaled_width * (1 - width_factor) // 2, height_offset + scaled_height // 3),
                (scaled_width * (1 + width_factor) // 2, height_offset + scaled_height // 3)
            ])

        return sprite

    def _create_rock_sprite(self) -> pygame.Surface:
        size = (self.game_config.rock_width * self.render_config.scale_factor,
                self.game_config.rock_height * self.render_config.scale_factor)
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        scaled_width = size[0]
        scaled_height = size[1]

        # Draw a polygon for the rock
        points = [
            (scaled_width * 0.2, scaled_height * 0.8),
            (0, scaled_height * 0.4),
            (scaled_width * 0.3, scaled_height * 0.2),
            (scaled_width * 0.7, scaled_height * 0.1),
            (scaled_width, scaled_height * 0.5),
            (scaled_width * 0.8, scaled_height)
        ]
        pygame.draw.polygon(sprite, self.render_config.rock_color, points)

        return sprite

    def render(self, state: GameState):
        """Render the current game state"""
        self.screen.fill(self.render_config.background_color)

        # Draw skier
        skier_pos = (
            int((state.skier_x - self.game_config.skier_width / 2) * self.render_config.scale_factor),
            int((self.game_config.skier_y - self.game_config.skier_height / 2) * self.render_config.scale_factor)
        )
        self.screen.blit(self.skier_sprite[state.skier_pos], skier_pos)

        # Draw flags
        for fx, fy in state.flags:
            flag_pos = (
                int((fx - self.game_config.flag_width / 2) * self.render_config.scale_factor),
                int((fy - self.game_config.flag_height / 2) * self.render_config.scale_factor)
            )
            self.screen.blit(self.flag_sprite, flag_pos)
            second_flag_pos = (flag_pos[0] + self.game_config.flag_distance * self.render_config.scale_factor, flag_pos[1])
            self.screen.blit(self.flag_sprite, second_flag_pos)

        for fx, fy in state.trees:
            tree_pos = (
                int((fx - self.game_config.tree_width / 2) * self.render_config.scale_factor),
                int((fy - self.game_config.tree_height / 2) * self.render_config.scale_factor)
            )
            self.screen.blit(self.tree_sprite, tree_pos)

        for fx, fy in state.rocks:
            rock_pos = (
                int((fx - self.game_config.rock_width / 2) * self.render_config.scale_factor),
                int((fy - self.game_config.rock_height / 2) * self.render_config.scale_factor)
            )
            self.screen.blit(self.rock_sprite, rock_pos)

        # Draw UI
        score_text = self.font.render(f"Score: {state.score}", True, self.render_config.text_color)
        time_text = self.font.render(f"Time: {state.time:.1f}", True, self.render_config.text_color)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (10, 40))

        if state.game_over:
            game_over_text = self.font.render("Game Over!", True, self.render_config.game_over_color)
            text_rect = game_over_text.get_rect(center=(
                self.game_config.screen_width * self.render_config.scale_factor // 2,
                self.game_config.screen_height * self.render_config.scale_factor // 2
            ))
            self.screen.blit(game_over_text, text_rect)

        pygame.display.flip()

    def close(self):
        """Clean up pygame resources"""
        pygame.quit()


# main.py
def main():
    # Create configurations
    game_config = GameConfig()
    render_config = RenderConfig()

    # Initialize game and renderer
    game = SkiingGameLogic(game_config)
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
        if keys[pygame.K_a]:
            action = LEFT
        elif keys[pygame.K_d]:
            action = RIGHT
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