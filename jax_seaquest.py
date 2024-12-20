from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import pygame

# Game Constants
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Colors
BACKGROUND_COLOR = (0, 0, 139)  # Dark blue for water
PLAYER_COLOR = (187, 187, 53)  # Yellow for player sub
DIVER_COLOR = (66, 72, 200)  # Pink for divers
SHARK_COLOR = (92, 186, 92)  # Green for sharks
ENEMY_SUB_COLOR = (170, 170, 170)  # Gray for enemy subs
OXYGEN_BAR_COLOR = (214, 214, 214)  # White for oxygen
SCORE_COLOR = (210, 210, 64)  # Score color
OXYGEN_TEXT_COLOR = (0, 0, 0)  # Black for oxygen text

# Object sizes and initial positions from RAM state
PLAYER_SIZE = (16, 11)  # Width, Height
DIVER_SIZE = (8, 11)
SHARK_SIZE = (8, 7)
ENEMY_SUB_SIZE = (8, 11)
MISSILE_SIZE = (8, 1)

PLAYER_START_X = 76
PLAYER_START_Y = 46

X_BORDERS = (0, 160)

# Maximum number of objects (from MAX_NB_OBJECTS)
MAX_DIVERS = 4
MAX_SHARKS = 12
MAX_SUBS = 12
MAX_ENEMY_MISSILES = 4
MAX_PLAYER_MISSILES = 1
MAX_SURFACE_SUBS = 1
MAX_COLLECTED_DIVERS = 6

# Define action space
NOOP = 0
FIRE = 1
UP = 2
RIGHT = 3
LEFT = 4
DOWN = 5
UPRIGHT = 6
UPLEFT = 7
DOWNRIGHT = 8
DOWNLEFT = 9
UPFIRE = 10
RIGHTFIRE = 11
LEFTFIRE = 12
DOWNFIRE = 13
UPRIGHTFIRE = 14
UPLEFTFIRE = 15
DOWNRIGHTFIRE = 16
DOWNLEFTFIRE = 17

SPAWN_POSTIONS_Y = jnp.array([69, 93, 117, 141])

class SpawnState(NamedTuple):
    difficulty: chex.Array  # Current difficulty level
    obstacle_pattern_indexes: chex.Array  # Current pattern indexes for each enemy
    obstacle_attributes: chex.Array  # Direction and type attributes for enemies
    spawn_timers: chex.Array  # Timers for spawning new enemies

# Game state container
class State(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_direction: chex.Array  # 0 for right, 1 for left
    oxygen: chex.Array
    divers_collected: chex.Array
    score: chex.Array
    lives: chex.Array
    spawn_state: SpawnState
    diver_positions: chex.Array  # (4, 2) array for divers
    shark_positions: chex.Array  # (12, 2) array for sharks
    sub_positions: chex.Array  # (12, 2) array for enemy subs
    enemy_missile_positions: chex.Array  # (4, 2) array for enemy missiles
    surface_sub_position: chex.Array  # (1, 2) array for surface submarine
    player_missile_position: chex.Array  # (1, 3) array for player missile (x, y, direction)
    collected_divers: chex.Array
    step_counter: chex.Array


# Spawn Constants from assembly analysis
INIT_OBSTACLE_PATTERNS = jnp.array([
    0x41,  # Difficulty 0: Left=4 (formation of 4), Right=1 (single)
    0x41,  # Difficulty 1: Same as 0
    0x63,  # Difficulty 2: Left=6 (formation of 6), Right=3 (formation of 3)
    0x63,  # Difficulty 3: Same as 2
    0x55,  # Difficulty 4: Left=5, Right=5
    0x55,  # Difficulty 5: Same as 4
    0x77,  # Difficulty 6: Left=7, Right=7
    0x77,  # Difficulty 7: Same as 6
])


class CarryState(NamedTuple):
    missile_pos: chex.Array
    shark_pos: chex.Array
    sub_pos: chex.Array
    score: chex.Array


def check_collision(pos1, size1, pos2, size2):
    """Check for collision between two rectangles given their positions and sizes"""
    return jnp.logical_and(
        jnp.logical_and(
            pos1[0] < pos2[0] + size2[0],
            pos1[0] + size1[0] > pos2[0]
        ),
        jnp.logical_and(
            pos1[1] < pos2[1] + size2[1],
            pos1[1] + size1[1] > pos2[1]
        )
    )


def check_missile_collisions(
        missile_pos: chex.Array,
        shark_positions: chex.Array,
        sub_positions: chex.Array,
        score: chex.Array
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Check for collisions between player missile and enemies"""

    # Create missile position array for collision check
    missile_rect_pos = jnp.array([missile_pos[0], missile_pos[1]])
    missile_active = missile_pos[2] != 0

    def check_enemy_collisions(enemy_idx, carry):
        # Unpack carry state using named tuple
        carry_state = CarryState(*carry)

        # Check shark collisions - only if missile is active
        shark_collision = jnp.logical_and(
            missile_active,
            check_collision(
                missile_rect_pos, MISSILE_SIZE,
                carry_state.shark_pos[enemy_idx], SHARK_SIZE
            )
        )

        # Check submarine collisions - only if missile is active
        sub_collision = jnp.logical_and(
            missile_active,
            check_collision(
                missile_rect_pos, MISSILE_SIZE,
                carry_state.sub_pos[enemy_idx], ENEMY_SUB_SIZE
            )
        )

        # Update positions and score - use where instead of if statements
        new_shark_pos = jnp.where(
            shark_collision,
            jnp.zeros_like(carry_state.shark_pos[enemy_idx]),
            carry_state.shark_pos[enemy_idx]
        )

        new_sub_pos = jnp.where(
            sub_collision,
            jnp.zeros_like(carry_state.sub_pos[enemy_idx]),
            carry_state.sub_pos[enemy_idx]
        )

        # Update score - sharks worth 20, subs worth 50
        score_increase = jnp.where(
            shark_collision,
            20,
            jnp.where(sub_collision, 50, 0)
        )

        # Remove missile if it hit anything
        new_missile_pos = jnp.where(
            jnp.logical_or(shark_collision, sub_collision),
            jnp.array([0., 0., 0.]),
            carry_state.missile_pos
        )

        return (new_missile_pos,
                carry_state.shark_pos.at[enemy_idx].set(new_shark_pos),
                carry_state.sub_pos.at[enemy_idx].set(new_sub_pos),
                carry_state.score + score_increase)

    # Initialize carry state
    init_carry = (missile_pos, shark_positions, sub_positions, score)

    # Always run the loop, but collisions only happen if missile is active
    return jax.lax.fori_loop(
        0, shark_positions.shape[0],
        check_enemy_collisions,
        init_carry
    )


def check_player_collision(player_x, player_y, submarine_list, shark_list, enemy_projectile_list) -> chex.Array:
    # check if the player has collided with any of the three given lists
    # the player is a 16x11 rectangle
    # the submarine is a 8x11 rectangle
    # the shark is a 8x7 rectangle
    # the missile is a 8x1 rectangle

    # check if the player has collided with any of the submarines -> jax compatibility
    submarine_collisions = jnp.any(
        check_collision(
            jnp.array([player_x, player_y]),
            PLAYER_SIZE,
            submarine_list,
            ENEMY_SUB_SIZE
        )
    )

    # check if the player has collided with any of the sharks -> jax compatibility
    shark_collisions = jnp.any(
        check_collision(
            jnp.array([player_x, player_y]),
            PLAYER_SIZE,
            shark_list,
            SHARK_SIZE
        )
    )

    # check if the player has collided with any of the enemy projectiles -> jax compatibility
    missile_collisions = jnp.any(
        check_collision(
            jnp.array([player_x, player_y]),
            PLAYER_SIZE,
            enemy_projectile_list,
            MISSILE_SIZE
        )
    )

    return jnp.any(jnp.array([submarine_collisions, shark_collisions, missile_collisions]))


def initialize_spawn_state() -> SpawnState:
    """Initialize the spawn state components"""
    return SpawnState(
        difficulty=jnp.array(0),
        obstacle_pattern_indexes=jnp.zeros(4, dtype=jnp.int32),
        obstacle_attributes=jnp.zeros(4, dtype=jnp.int32),
        spawn_timers=jnp.zeros(4, dtype=jnp.int32)
    )


def get_spawn_pattern(difficulty: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Get left and right spawn patterns for current difficulty"""
    pattern = INIT_OBSTACLE_PATTERNS[difficulty & 0x07]
    left_pattern = (pattern >> 4) & 0x0F  # Upper 4 bits
    right_pattern = pattern & 0x0F  # Lower 4 bits
    return left_pattern, right_pattern


def get_spawn_position(moving_left: bool, slot: chex.Array) -> chex.Array:
    """Get spawn position based on movement direction and slot number"""
    # Y positions are spaced vertically in the playable area
    # The playable area is between the surface and sea floor

    base_y = jnp.array(SPAWN_POSTIONS_Y[slot])

    x_pos = jnp.where(moving_left,
                      jnp.array(160 + 8),
                      jnp.array(-41))

    return jnp.array([x_pos, base_y])


def update_enemy_spawns(spawn_state: SpawnState,
                        shark_positions: chex.Array,
                        sub_positions: chex.Array,
                        step_counter: chex.Array) -> Tuple[SpawnState, chex.Array, chex.Array]:
    """Update enemy spawn positions and patterns"""
    left_pattern, right_pattern = get_spawn_pattern(spawn_state.difficulty)

    # For each position slot (0-3)
    def update_slot(i, carry):
        spawn_state, shark_pos, sub_pos = carry

        # Check if slot is empty (position off screen or timer expired)
        is_empty = jnp.logical_or(
            jnp.all(shark_pos[i] <= 0),
            jnp.all(sub_pos[i] <= 0)
        )

        # Determine spawn direction and pattern
        should_spawn_left = (step_counter + i) % 2 == 0
        pattern = jnp.where(should_spawn_left, left_pattern, right_pattern)

        # Get spawn position including vertical position based on slot
        new_pos = get_spawn_position(should_spawn_left, jnp.array(i))

        # Update positions and patterns if should spawn
        should_be_sub = pattern & 0x04  # Check if this should be a submarine

        new_shark_pos = jnp.where(
            jnp.logical_and(is_empty, ~should_be_sub),
            new_pos,
            shark_pos[i]
        )

        new_sub_pos = jnp.where(
            jnp.logical_and(is_empty, should_be_sub),
            new_pos,
            sub_pos[i]
        )

        return spawn_state, \
            shark_pos.at[i].set(new_shark_pos), \
            sub_pos.at[i].set(new_sub_pos)

    # Update each slot
    spawn_state, new_shark_pos, new_sub_pos = jax.lax.fori_loop(
        0, 4, update_slot, (spawn_state, shark_positions, sub_positions)
    )

    return spawn_state, new_shark_pos, new_sub_pos


def step_enemy_movement(spawn_state: SpawnState,
                        shark_positions: chex.Array,
                        sub_positions: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Update enemy positions based on their patterns"""

    def move_enemy(pos, moving_left):
        # Move 1 unit per frame in appropriate direction
        velocity = jnp.where(moving_left, -1, 1)
        new_pos = pos + jnp.array([velocity, 0])

        # Check bounds and zero out if off screen
        out_of_bounds = jnp.logical_or(new_pos[0] < -41, new_pos[0] > 168)
        return jnp.where(out_of_bounds, jnp.zeros_like(new_pos), new_pos)

    # Move each enemy based on their pattern
    new_shark_positions = jnp.stack([
        move_enemy(pos, (spawn_state.obstacle_attributes[i] & 0x08) != 0)
        for i, pos in enumerate(shark_positions)
    ])

    new_sub_positions = jnp.stack([
        move_enemy(pos, (spawn_state.obstacle_attributes[i] & 0x08) != 0)
        for i, pos in enumerate(sub_positions)
    ])

    return new_shark_positions, new_sub_positions


def spawn_step(state: State, spawn_state: SpawnState, shark_positions: chex.Array, sub_positions: chex.Array) -> Tuple[SpawnState, chex.Array, chex.Array]:
    """Main spawn handling function to be called in game step"""
    # Update spawns
    spawn_state, new_shark_positions, new_sub_positions = update_enemy_spawns(
        spawn_state,
        shark_positions,
        sub_positions,
        state.step_counter
    )

    # Move existing enemies
    new_shark_positions, new_sub_positions = step_enemy_movement(
        spawn_state,
        new_shark_positions,
        new_sub_positions
    )

    return spawn_state, new_shark_positions, new_sub_positions

def player_missile_step(state: State, curr_player_x, curr_player_y, action: chex.Array) -> chex.Array:
    # check if the player shot this frame
    fire = jnp.any(jnp.array([action == FIRE, action == UPRIGHTFIRE, action == UPLEFTFIRE, action == DOWNFIRE, action == DOWNRIGHTFIRE, action == DOWNLEFTFIRE, action == RIGHTFIRE, action == LEFTFIRE, action == UPFIRE]))

    # IMPORTANT: do not change the order of this check, since the missile does not move in its first frame!!
    # also check if there is currently a missile in frame by checking if the player_missile_position is empty
    missile_exists = state.player_missile_position[2] != 0

    # if the player shot and there is no missile in frame, then we can shoot a missile
    # the missile y is the current player y position
    # the missile x is either player x + 3 if facing left or player x + 13 if facing right
    new_missile = jnp.where(
        jnp.logical_and(fire, ~missile_exists),
        jnp.where(
            state.player_direction == -1,
            jnp.array([curr_player_x + 3, curr_player_y, -1]),
            jnp.array([curr_player_x + 13, curr_player_y, 1])
        ),
        state.player_missile_position
    )

    # if a missile is in frame and exists, we move the missile further in the specified direction (5 per tick), also always put the missile at the current player y position
    new_missile = jnp.where(
        missile_exists,
        jnp.array([new_missile[0] + new_missile[2]*5, curr_player_y, new_missile[2]]),
        new_missile
    )

    # check if the new positions are still in bounds
    new_missile = jnp.where(
        new_missile[0] < X_BORDERS[0],
        jnp.array([0, 0, 0]),
        jnp.where(
            new_missile[0] > X_BORDERS[1],
            jnp.array([0, 0, 0]),
            new_missile
        )
    )

    return new_missile

def player_step(state: State, action: chex.Array) -> tuple[chex.Array, chex.Array, chex.Array]:
    # implement all the possible movement directions for the player, the mapping is:
    # anything with left in it, add -1 to the x position
    # anything with right in it, add 1 to the x position
    # anything with up in it, add -1 to the y position
    # anything with down in it, add 1 to the y position
    up = jnp.any(jnp.array([action == UP, action == UPRIGHT, action == UPLEFT, action == UPFIRE, action == UPRIGHTFIRE, action == UPLEFTFIRE]))
    down = jnp.any(jnp.array([action == DOWN, action == DOWNRIGHT, action == DOWNLEFT, action == DOWNFIRE, action == DOWNRIGHTFIRE, action == DOWNLEFTFIRE]))
    left = jnp.any(jnp.array([action == LEFT, action == UPLEFT, action == DOWNLEFT, action == LEFTFIRE, action == UPLEFTFIRE, action == DOWNLEFTFIRE]))
    right = jnp.any(jnp.array([action == RIGHT, action == UPRIGHT, action == DOWNRIGHT, action == RIGHTFIRE, action == UPRIGHTFIRE, action == DOWNRIGHTFIRE]))

    player_x = jnp.where(
        right,
        state.player_x + 1,
        jnp.where(
            left,
            state.player_x - 1,
            state.player_x
        )
    )

    player_y = jnp.where(
        down,
        state.player_y + 1,
        jnp.where(
            up,
            state.player_y - 1,
            state.player_y
        )
    )

    # set the direction according to the movement
    player_direction = jnp.where(
        right,
        1,
        jnp.where(
            left,
            -1,
            state.player_direction
        )
    )

    return player_x, player_y, player_direction

class Game:
    def __init__(self, frameskip: int = 1):
        self.frameskip = frameskip
        pass

    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> State:
        """Initialize game state"""
        return State(
            player_x=jnp.array(PLAYER_START_X),
            player_y=jnp.array(PLAYER_START_Y),
            player_direction=jnp.array(0),
            oxygen=jnp.array(0),  # Full oxygen
            divers_collected=jnp.array(0),
            score=jnp.array(0),
            lives=jnp.array(3),
            spawn_state=initialize_spawn_state(),
            diver_positions=jnp.zeros((MAX_DIVERS, 2)),  # 4 divers
            shark_positions=jnp.zeros((MAX_SHARKS, 2)),  # 12 sharks
            sub_positions=jnp.zeros((MAX_SUBS, 2)),  # 12 subs
            enemy_missile_positions=jnp.zeros((MAX_ENEMY_MISSILES, 2)),  # 4 missiles
            surface_sub_position=jnp.zeros((MAX_SURFACE_SUBS, 2)),  # 1 surface sub
            player_missile_position=jnp.zeros(3),  # x,y,direction
            collected_divers=jnp.zeros(MAX_COLLECTED_DIVERS),
            step_counter=jnp.array(0)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        player_x, player_y, player_direction = player_step(state, action)

        player_missile_position = player_missile_step(state, player_x, player_y, action)

        # Check missile collisions
        player_missile_position, new_shark_positions, new_sub_positions, new_score = check_missile_collisions(
            player_missile_position,
            state.shark_positions,
            state.sub_positions,
            state.score
        )

        new_spawn_state, new_shark_positions, new_sub_positions = spawn_step(
            state,
            state.spawn_state,
            new_shark_positions,
            new_sub_positions
        )

        # check if the player is under the water surface, if so, decrease the oxygen level by 1 every 32 ticks
        # if the player is above the water surface, increase the oxygen level by 1 every 2 ticks until it reaches 64

        # check if oxygen should decrease or increase
        decrease_ox = player_y > 53
        increase_ox = player_y <= 53

        # decrease the oxygen level by 1 every 32 ticks
        new_oxygen = jnp.where(
            decrease_ox,
            jnp.where(
                state.step_counter % 32 == 0,
                state.oxygen - 1,
                state.oxygen
            ),
            state.oxygen
        )

        # increase the oxygen level by 1 every 2 ticks until it reaches 64
        new_oxygen = jnp.where(
            increase_ox,
            jnp.where(
                state.oxygen < 64,
                jnp.where(
                    state.step_counter % 2 == 0,
                    state.oxygen + 1,
                    state.oxygen
                ),
                state.oxygen
            ),
            new_oxygen
        )

        oxygen_depleted = new_oxygen <= 0

        # 36 - 128 -> 92 ticks wait
        # 128 ticks oxygen replenish?
        # oxygen decreases by 1 every 32 ticks

        # check if the player has collided with any of the enemies
        player_collision = check_player_collision(player_x, player_y, new_sub_positions, new_shark_positions, state.enemy_missile_positions)

        # Return unchanged state for now
        return State(
            player_x=player_x,
            player_y=player_y,
            player_direction=player_direction,
            oxygen=new_oxygen,  # Full oxygen
            divers_collected=jnp.array(0),
            score=new_score,
            lives=jnp.array(3),
            spawn_state=new_spawn_state,
            diver_positions=jnp.zeros((MAX_DIVERS, 2)),  # 4 divers
            shark_positions=new_shark_positions,  # 12 sharks
            sub_positions=new_sub_positions,  # 12 subs
            enemy_missile_positions=jnp.zeros((MAX_ENEMY_MISSILES, 2)),  # 4 missiles
            surface_sub_position=jnp.zeros((MAX_SURFACE_SUBS, 2)),  # 1 surface sub
            player_missile_position=player_missile_position,  # 1 player missile
            collected_divers=jnp.zeros(MAX_COLLECTED_DIVERS),
            step_counter=state.step_counter + 1
        )


class Renderer:
    def __init__(self):
        """Initialize the renderer"""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Seaquest")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def draw_water_gradient(self):
        """Draw water surface gradient effect"""
        surface_gradient = pygame.Surface((WINDOW_WIDTH, 30))
        for i in range(30):
            alpha = 255 - (i * 8)
            color = (*BACKGROUND_COLOR[:2], min(255, BACKGROUND_COLOR[2] + 50))
            pygame.draw.line(surface_gradient, color, (0, i), (WINDOW_WIDTH, i))
        self.screen.blit(surface_gradient, (0, 0))

    def draw_oxygen_bar(self, oxygen_value):
        """Draw oxygen bar and text"""
        # Draw "OXYGEN" text
        text = self.font.render("OXYGEN", True, OXYGEN_TEXT_COLOR)
        self.screen.blit(text, (15 * 3, 170 * 3))

        # Draw oxygen bar
        oxygen_width = int((float(oxygen_value) / 63.0) * 180)
        oxygen_rect = pygame.Rect(49 * 3, 170 * 3, oxygen_width, 15)
        pygame.draw.rect(self.screen, OXYGEN_BAR_COLOR, oxygen_rect)

    def draw_score_and_lives(self, score, lives):
        """Draw score and lives counter"""
        score_text = self.font.render(str(int(score)), True, SCORE_COLOR)
        lives_text = self.font.render(f"Lives: {int(lives)}", True, SCORE_COLOR)
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (WINDOW_WIDTH - 100, 10))

    def draw_enemies(self, shark_positions, sub_positions):
        """Draw sharks and submarines"""
        # Draw sharks
        for pos in shark_positions:
            if pos[0] > 0:  # Only draw if x position is valid
                shark_rect = pygame.Rect(
                    int(pos[0]) * 3, int(pos[1]) * 3,
                    SHARK_SIZE[0] * 3, SHARK_SIZE[1] * 3
                )
                pygame.draw.rect(self.screen, SHARK_COLOR, shark_rect)

        # Draw submarines
        for pos in sub_positions:
            if pos[0] > 0:  # Only draw if x position is valid
                sub_rect = pygame.Rect(
                    int(pos[0]) * 3, int(pos[1]) * 3,
                    ENEMY_SUB_SIZE[0] * 3, ENEMY_SUB_SIZE[1] * 3
                )
                pygame.draw.rect(self.screen, ENEMY_SUB_COLOR, sub_rect)

    def draw_divers(self, diver_positions):
        """Draw divers"""
        for pos in diver_positions:
            if pos[0] > 0:  # Only draw if x position is valid
                diver_rect = pygame.Rect(
                    int(pos[0]) * 3, int(pos[1]) * 3,
                    DIVER_SIZE[0] * 3, DIVER_SIZE[1] * 3
                )
                pygame.draw.rect(self.screen, DIVER_COLOR, diver_rect)

    def draw_player(self, x, y):
        """Draw player submarine"""
        player_rect = pygame.Rect(
            int(x) * 3, int(y) * 3,
            PLAYER_SIZE[0] * 3, PLAYER_SIZE[1] * 3
        )
        pygame.draw.rect(self.screen, PLAYER_COLOR, player_rect)

    def draw_missiles(self, missile_positions):

        # check if there is only a single missile (i.e. not multiple 3 element arrays)
        if len(missile_positions.shape) == 1:
            missile_positions = jnp.expand_dims(missile_positions, axis=0)

        """Draw missiles"""
        for pos in missile_positions:
            if pos[0] > 0:  # Only draw if x position is valid
                missile_rect = pygame.Rect(
                    int(pos[0]) * 3, int(pos[1]) * 3,
                    MISSILE_SIZE[0] * 3, MISSILE_SIZE[1] * 3
                )
                pygame.draw.rect(self.screen, PLAYER_COLOR, missile_rect)

    def render(self, state: State):
        """Main render method that draws everything"""
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)

        # Draw background effects
        self.draw_water_gradient()

        # Draw game objects
        self.draw_divers(state.diver_positions)
        self.draw_enemies(state.shark_positions, state.sub_positions)
        self.draw_missiles(state.player_missile_position)
        self.draw_missiles(state.enemy_missile_positions)
        self.draw_player(state.player_x, state.player_y)

        # Draw HUD elements
        self.draw_oxygen_bar(state.oxygen)
        self.draw_score_and_lives(state.score, state.lives)

        # Update display
        pygame.display.flip()
        self.clock.tick(60)


def get_human_action() -> chex.Array:
    """Get human action from keyboard with support for diagonal movement and combined fire"""
    keys = pygame.key.get_pressed()
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    fire = keys[pygame.K_SPACE]

    # Diagonal movements with fire
    if up and right and fire:
        return jnp.array(UPRIGHTFIRE)
    if up and left and fire:
        return jnp.array(UPLEFTFIRE)
    if down and right and fire:
        return jnp.array(DOWNRIGHTFIRE)
    if down and left and fire:
        return jnp.array(DOWNLEFTFIRE)

    # Cardinal directions with fire
    if up and fire:
        return jnp.array(UPFIRE)
    if down and fire:
        return jnp.array(DOWNFIRE)
    if left and fire:
        return jnp.array(LEFTFIRE)
    if right and fire:
        return jnp.array(RIGHTFIRE)

    # Diagonal movements
    if up and right:
        return jnp.array(UPRIGHT)
    if up and left:
        return jnp.array(UPLEFT)
    if down and right:
        return jnp.array(DOWNRIGHT)
    if down and left:
        return jnp.array(DOWNLEFT)

    # Cardinal directions
    if up:
        return jnp.array(UP)
    if down:
        return jnp.array(DOWN)
    if left:
        return jnp.array(LEFT)
    if right:
        return jnp.array(RIGHT)
    if fire:
        return jnp.array(FIRE)

    return jnp.array(NOOP)


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
            elif event.type == pygame.KEYDOWN or (event.type == pygame.KEYUP and event.key == pygame.K_n):
                if event.key == pygame.K_n and frame_by_frame:
                    if counter % frameskip == 0:
                        action = get_human_action()
                        curr_state = jitted_step(curr_state, action)

        if not frame_by_frame:
            if counter % frameskip == 0:
                action = get_human_action()
                curr_state = jitted_step(curr_state, action)

        renderer.render(curr_state)
        counter += 1
        renderer.clock.tick(60)

    pygame.quit()