import sys
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import pygame
import atraJaxis as aj
import numpy as np


# Game Constants
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

WIDTH = 160
HEIGHT = 210
SCALING_FACTOR = 3

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
PLAYER_BOUNDS = (21, 134), (46, 141)

# Maximum number of objects (from MAX_NB_OBJECTS)
MAX_DIVERS = 4
MAX_SHARKS = 4
MAX_SUBS = 4
MAX_ENEMY_MISSILES = 4
MAX_PLAYER_TORPS = 1
MAX_SURFACE_SUBS = 1
MAX_COLLECTED_DIVERS = 6

# define object orientations
FACE_LEFT = -1
FACE_RIGHT = 1

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

SPAWN_POSITIONS_Y = jnp.array([71, 95, 119, 139]) # submarines at y=69?
SUBMARINE_Y_OFFSET = 2
ENEMY_MISSILE_Y = jnp.array([73, 97, 121, 141]) # missile x = submarine.x + 4
DIVER_SPAWN_POSITIONS = jnp.array([69, 93, 117, 141])

MISSILE_SPAWN_POSITIONS = jnp.array([39, 126])  # Right, Left

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
    diver_positions: chex.Array  # (4, 3) array for divers
    shark_positions: chex.Array  # (4, 3) array for sharks
    sub_positions: chex.Array  # (4, 3) array for enemy subs
    enemy_missile_positions: chex.Array  # (4, 3) array for enemy missiles
    surface_sub_position: chex.Array  # (1, 2) array for surface submarine
    player_missile_position: chex.Array  # (1, 3) array for player missile (x, y, direction)
    step_counter: chex.Array
    just_surfaced: chex.Array  # Flag for tracking actual surfacing moment


class CarryState(NamedTuple):
    missile_pos: chex.Array
    shark_pos: chex.Array
    sub_pos: chex.Array
    score: chex.Array

# RENDER CONSTANTS
def load_sprites():
    # Load sprites - no padding needed for background since it's already full size
    bg1 = jnp.flip(jnp.array(aj.loadFrame('sprites/seaquest/bg/1.npy')), axis=0)
    pl_sub1 = jnp.transpose(jnp.array(aj.loadFrame('sprites/seaquest/player_sub/1.npy')), (1, 0, 2))
    pl_sub2 = jnp.transpose(jnp.array(aj.loadFrame('sprites/seaquest/player_sub/2.npy')), (1, 0, 2))
    pl_sub3 = jnp.transpose(jnp.array(aj.loadFrame('sprites/seaquest/player_sub/3.npy')), (1, 0, 2))
    diver1 = jnp.flip(jnp.flip(jnp.array(aj.loadFrame('sprites/seaquest/diver/1.npy')), axis=0), axis=1)
    diver2 = jnp.flip(jnp.flip(jnp.array(aj.loadFrame('sprites/seaquest/diver/2.npy')), axis=0), axis=1)

    # Only pad sprites of same type to match each other's dimensions
    def pad_to_match(sprites):
        max_height = max(sprite.shape[0] for sprite in sprites)
        max_width = max(sprite.shape[1] for sprite in sprites)

        def pad_sprite(sprite):
            pad_height = max_height - sprite.shape[0]
            pad_width = max_width - sprite.shape[1]
            return jnp.pad(sprite,
                          ((0, pad_height), (0, pad_width), (0, 0)),
                          mode='constant',
                          constant_values=0)

        return [pad_sprite(sprite) for sprite in sprites]

    # Pad player submarine sprites to match each other
    pl_sub_sprites = pad_to_match([pl_sub1, pl_sub2, pl_sub3])

    # Pad diver sprites to match each other
    diver_sprites = pad_to_match([diver1, diver2])

    # Background sprite (no padding needed)
    SPRITE_BG = jnp.expand_dims(bg1, axis=0)

    # Player submarine sprites
    SPRITE_PL_SUB = jnp.concatenate([
        jnp.repeat(pl_sub_sprites[0][None], 4, axis=0),
        jnp.repeat(pl_sub_sprites[1][None], 4, axis=0),
        jnp.repeat(pl_sub_sprites[2][None], 4, axis=0)
    ])

    # Diver sprites
    SPRITE_DIVER = jnp.concatenate([
        jnp.repeat(diver_sprites[0][None], 16, axis=0),
        jnp.repeat(diver_sprites[1][None], 4, axis=0)
    ])

    return SPRITE_BG, SPRITE_PL_SUB, SPRITE_DIVER


# Load sprites once at module level
SPRITE_BG, SPRITE_PL_SUB, SPRITE_DIVER = load_sprites()


# TODO: remove, for debugging purposes only
def print(arg1, arg2=None):
    jax.debug.print("{x}: {y}", x=arg1, y=arg2)

# TODO: there is some pixel overlap in the OG implementation, analyse and implement
def check_collision(pos1, size1, pos2, size2):
    """
    Check for collision between rectangles.
    Handles both single position and array of positions for pos2.

    Args:
        pos1: Single position [x,y]
        size1: Single size (width,height)
        pos2: Single position [x,y] or array of positions
        size2: Single size (width,height)
    """
    # Calculate edges for rectangle 1
    rect1_left = pos1[0]
    rect1_right = pos1[0] + size1[0]
    rect1_top = pos1[1]
    rect1_bottom = pos1[1] + size1[1]

    # Handle both single positions and arrays
    is_array = len(pos2.shape) > 1

    if is_array:
        # Array of positions
        rect2_left = pos2[:, 0]
        rect2_right = pos2[:, 0] + size2[0]
        rect2_top = pos2[:, 1]
        rect2_bottom = pos2[:, 1] + size2[1]
    else:
        # Single position
        rect2_left = pos2[0]
        rect2_right = pos2[0] + size2[0]
        rect2_top = pos2[1]
        rect2_bottom = pos2[1] + size2[1]

    # Check overlap
    horizontal_overlaps = jnp.logical_and(
        rect1_left < rect2_right,
        rect1_right > rect2_left
    )

    vertical_overlaps = jnp.logical_and(
        rect1_top < rect2_bottom,
        rect1_bottom > rect2_top
    )

    # Combine checks
    collisions = jnp.logical_and(horizontal_overlaps, vertical_overlaps)

    # Return single boolean for any collision
    return jnp.any(collisions)


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

    # check if the player has collided with any of the submarines
    submarine_collisions = jnp.any(
        check_collision(
            jnp.array([player_x, player_y]),
            PLAYER_SIZE,
            submarine_list,
            ENEMY_SUB_SIZE
        )
    )

    # check if the player has collided with any of the sharks
    shark_collisions = jnp.any(
        check_collision(
            jnp.array([player_x, player_y]),
            PLAYER_SIZE,
            shark_list,
            SHARK_SIZE
        )
    )

    # check if the player has collided with any of the enemy projectiles
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
    """Initialize the spawn state with proper patterns and timers"""
    return SpawnState(
        difficulty=jnp.array(0),
        # Initialize with cycling patterns for each slot
        obstacle_pattern_indexes=jnp.array([0, 1, 2, 3], dtype=jnp.int32),
        # Initial attributes: alternate directions
        obstacle_attributes=jnp.array([0x08, 0x00, 0x08, 0x00], dtype=jnp.int32),
        # Staggered initial spawn timers (more spread out)
        spawn_timers=jnp.array([0, 30, 60, 90], dtype=jnp.int32)
    )


def get_spawn_position(moving_left: bool, slot: chex.Array) -> chex.Array:
    """Get spawn position based on movement direction and slot number"""
    base_y = jnp.array(SPAWN_POSITIONS_Y[slot])
    x_pos = jnp.where(moving_left,
                      jnp.array(165),  # Start right if moving left
                      jnp.array(0))    # Start left if moving right
    direction = jnp.where(moving_left, -1, 1)  # -1 for left, 1 for right
    return jnp.array([x_pos, base_y, direction])


def is_slot_empty(pos: chex.Array) -> chex.Array:
    """Check if a position slot is empty (0,0 position)"""
    return jnp.logical_and(
        pos[0] == 0,  # Check if x is 0
        pos[1] == 0   # Check if y is 0
    )


def update_enemy_spawns(spawn_state: SpawnState,
                        shark_positions: chex.Array,
                        sub_positions: chex.Array,
                        step_counter: chex.Array,
                        rng: chex.PRNGKey) -> Tuple[SpawnState, chex.Array, chex.Array]:
    """Update enemy spawn positions using proper pattern detection"""

    def update_slot(i, carry):
        spawn_state, shark_pos, sub_pos, rng = carry

        # Split RNG key for this iteration
        rng, sub_rng = jax.random.split(rng)

        # Rest of the checks remain same
        shark_empty = is_slot_empty(shark_pos[i])
        sub_empty = is_slot_empty(sub_pos[i])
        slot_empty = jnp.logical_and(shark_empty, sub_empty)
        spawn_timer = spawn_state.spawn_timers[i]
        pattern_idx = spawn_state.obstacle_pattern_indexes[i]
        can_spawn = spawn_timer <= 0
        should_spawn = jnp.logical_and(slot_empty, can_spawn)

        # Use the split RNG key for bernoulli
        should_be_sub = jax.random.bernoulli(sub_rng, 0.5)

        # Rest of the logic remains same
        new_timer = jnp.where(spawn_timer > 0,
                              spawn_timer - 1,
                              jnp.where(should_spawn,
                                        320,  # Reset timer when spawning new enemy
                                        spawn_timer))

        should_spawn_left = i % 2 == 0
        spawn_pos = get_spawn_position(should_spawn_left, jnp.array(i))

        new_shark_pos = jnp.where(
            jnp.logical_and(should_spawn, ~should_be_sub),
            spawn_pos,
            shark_pos[i]
        )

        new_sub_pos = jnp.where(
            jnp.logical_and(should_spawn, should_be_sub),
            spawn_pos,
            sub_pos[i]
        )

        new_spawn_state = SpawnState(
            difficulty=spawn_state.difficulty,
            obstacle_pattern_indexes=spawn_state.obstacle_pattern_indexes.at[i].set(pattern_idx),
            obstacle_attributes=spawn_state.obstacle_attributes,
            spawn_timers=spawn_state.spawn_timers.at[i].set(new_timer)
        )

        return new_spawn_state, shark_pos.at[i].set(new_shark_pos), sub_pos.at[i].set(new_sub_pos), rng

    # Update each slot (but not for the first 277 steps)
    return jax.lax.cond(
        step_counter < 277,
        lambda _: (spawn_state, shark_positions, sub_positions),
        lambda _: jax.lax.fori_loop(0, shark_positions.shape[0],
                                    update_slot,
                                    (spawn_state, shark_positions, sub_positions, rng))[:-1],
        # Drop the rng from result
        operand=None
    )


def step_enemy_movement(spawn_state: SpawnState,
                        shark_positions: chex.Array,
                        sub_positions: chex.Array,
                        step_counter: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """Update enemy positions based on their patterns"""

    def should_move_x(step_counter):
        # X movement logic remains the same as before
        cycle = (step_counter // 5) % 3
        frame_in_cycle = step_counter % 5

        normal_cycle = jnp.logical_or(frame_in_cycle == 0, frame_in_cycle == 3)
        short_cycle = jnp.logical_or(frame_in_cycle == 0, frame_in_cycle == 2)

        return jnp.where(cycle == 2, short_cycle, normal_cycle)

    def get_shark_offset(step_counter):
        phase = step_counter // 4
        cycle_position = phase % 32

        raw_offset = jnp.where(
            cycle_position < 16,
            cycle_position // 2,  # 0->7
            7 - (cycle_position - 16) // 2  # 7->0
        )

        return raw_offset - 4

    def move_enemy(pos, moving_left, is_shark, slot_idx):
        is_active = ~is_slot_empty(pos)

        # X movement
        should_move_now = should_move_x(step_counter)
        velocity_x = jnp.where(moving_left, -1, 1)
        movement_x = jnp.where(should_move_now, velocity_x, 0)

        # Base Y position comes from spawn positions
        base_y = SPAWN_POSITIONS_Y[slot_idx]

        # Calculate Y position
        y_position = jnp.where(
            is_shark,
            base_y + get_shark_offset(step_counter),
            # Submarines are 2 pixels higher than their base position
            base_y - SUBMARINE_Y_OFFSET
        )

        # Apply movements
        new_pos = jnp.where(
            is_active,
            jnp.array([pos[0] + movement_x, y_position, pos[2]]),
            pos
        )

        # Check bounds
        out_of_bounds = jnp.logical_or(new_pos[0] <= -8, new_pos[0] >= 168)
        return jnp.where(out_of_bounds, jnp.zeros(3), new_pos)

    # Move enemies
    # Each slot belongs to a lane - we need the actual lane index (0-3)
    new_shark_positions = jnp.stack([
        move_enemy(pos, i % 2 == 0, True, i // 1)  # Integer division by 1 to get slot index
        for i, pos in enumerate(shark_positions)
    ])

    new_sub_positions = jnp.stack([
        move_enemy(pos, i % 2 == 0, False, i // 1)
        for i, pos in enumerate(sub_positions)
    ])

    return new_shark_positions, new_sub_positions


def spawn_divers(spawn_state: SpawnState, diver_positions: chex.Array, shark_positions: chex.Array,
                 sub_positions: chex.Array, step_counter: chex.Array, force_spawn: bool = True) -> chex.Array:
    """Spawn divers according to patterns with directional awareness of enemies in their lane.
    Will not spawn divers in lanes with submarines.

    Args:
        spawn_state: Current spawn state
        diver_positions: Current diver positions
        shark_positions: Current shark positions (for lane direction)
        sub_positions: Current sub positions (for lane direction)
        step_counter: Current step counter
        force_spawn: If True, forces a diver spawn in each empty lane. For testing.
    """

    def spawn_diver(i, carry):
        # Unpack the carry tuple
        left, positions = carry

        # Get current diver position and enemy positions for this lane
        diver_pos = positions[i]
        shark_in_lane = shark_positions[i]
        sub_in_lane = sub_positions[i]

        # Check if a diver exists in this slot
        diver_exists = diver_pos[2] != 0

        # Check if lane is blocked by submarine
        lane_has_sub = sub_in_lane[2] != 0

        # Check if we should spawn based on force_spawn flag, existing diver, and lane availability
        should_spawn = jnp.logical_and(
            force_spawn,
            jnp.logical_and(~diver_exists, ~lane_has_sub)  # Only spawn if no sub and no existing diver
        )

        # Get direction from enemies in the lane (primarily sharks since subs block spawning)
        shark_active = shark_in_lane[2] != 0
        enemy_dir = jnp.where(
            shark_active,
            shark_in_lane[2],
            # Default direction based on spawn side if no enemies
            jnp.where(left, 1, -1)
        )

        # Set spawn position based on enemy direction
        x_pos = jnp.where(enemy_dir > 0, 0, 160)  # Opposite side from movement direction

        # Spawn diver with direction matching enemies
        new_diver = jnp.where(
            should_spawn,
            jnp.array([x_pos, DIVER_SPAWN_POSITIONS[i], enemy_dir]),
            diver_pos
        )

        # Return updated carry tuple
        return (left, positions.at[i].set(new_diver))

    # Initialize RNG (keeping for pattern consistency)
    rng = jax.random.PRNGKey(42)
    rng, diver_rng = jax.random.split(rng)
    left = jax.random.bernoulli(diver_rng, 0.5)

    # Initialize carry tuple
    initial_carry = (left, diver_positions)

    # Update all diver positions
    _, new_diver_positions = jax.lax.fori_loop(
        0, diver_positions.shape[0],
        spawn_diver,
        initial_carry
    )

    return new_diver_positions


def step_diver_movement(diver_positions: chex.Array, shark_positions: chex.Array,
                        state_player_x: chex.Array, state_player_y: chex.Array,
                        state_divers_collected: chex.Array,
                        step_counter: chex.Array) -> tuple[chex.Array, chex.Array]:
    """Move divers according to their pattern and handle collisions.
    Returns updated diver positions and number of collected divers.
    """

    def move_single_diver(i, carry):
        # Unpack carry state - (positions, collected_count)
        positions, collected = carry
        diver_pos = positions[i]

        # Only process active divers (direction != 0)
        is_active = diver_pos[2] != 0

        # Check for collision with player first if diver is active
        player_collision = jnp.logical_and(
            is_active,
            check_collision(
                jnp.array([state_player_x, state_player_y]),
                PLAYER_SIZE,
                jnp.array([diver_pos[0], diver_pos[1]]),
                DIVER_SIZE
            )
        )

        # Only collect if we haven't reached max divers
        can_collect = state_divers_collected < 6
        should_collect = jnp.logical_and(player_collision, can_collect)

        # Calculate the cycle position within the pattern
        cycle = step_counter % 14  # Total cycle length: 4 + 1 + 4 + 1 + 5 + 1 = 14

        # Get shark in the same lane for collision check
        shark_lane_pos = shark_positions[i]
        shark_collision = jnp.logical_and(
            is_active,
            check_collision(
                jnp.array([shark_lane_pos[0], shark_lane_pos[1]]),
                SHARK_SIZE,
                jnp.array([diver_pos[0], diver_pos[1]]),
                DIVER_SIZE
            )
        )

        # If colliding with shark, match shark's position and direction
        movement_x = jnp.where(
            shark_collision,
            shark_lane_pos[2],  # Use shark's direction/speed
            diver_pos[2]  # Use diver's normal direction
        )

        # Determine if we should move in this frame when not pushed by shark
        should_move = jnp.logical_or(
            cycle == 4,  # First move after 4 frames
            jnp.logical_or(
                cycle == 9,  # Second move after 4 more frames
                cycle == 13  # Third move after 5 more frames
            )
        )

        # Calculate new position - move every frame if pushed by shark, otherwise follow pattern
        new_x = jnp.where(
            shark_collision,
            diver_pos[0] + movement_x,  # Move with shark
            jnp.where(
                should_move,
                diver_pos[0] + movement_x,  # Normal movement
                diver_pos[0]  # Stay still
            )
        )

        # Check bounds
        out_of_bounds = jnp.logical_or(new_x <= -8, new_x >= 168)

        # Create new position array - handle collection and bounds
        new_pos = jnp.where(
            jnp.logical_or(out_of_bounds, should_collect),
            jnp.zeros(3),  # Reset if out of bounds or collected
            jnp.array([new_x, diver_pos[1], diver_pos[2]])
        )

        # Update collection count if collected
        new_collected = collected + jnp.where(should_collect, 1, 0)

        # Update the diver position and collection count
        return (positions.at[i].set(new_pos), new_collected)

    # Update all diver positions and track collections
    initial_carry = (diver_positions, state_divers_collected)
    final_positions, final_collected = jax.lax.fori_loop(
        0, diver_positions.shape[0],
        move_single_diver,
        initial_carry
    )

    return final_positions, final_collected

def spawn_step(state, spawn_state: SpawnState, shark_positions: chex.Array, sub_positions: chex.Array, diver_positions: chex.Array) -> Tuple[SpawnState, chex.Array, chex.Array, chex.Array]:
    """Main spawn handling function to be called in game step"""
    # Move existing enemies
    new_shark_positions, new_sub_positions = step_enemy_movement(
        spawn_state,
        shark_positions,
        sub_positions,
        state.step_counter
    )

    # random number generator TODO: remove, there is a fixed pattern somewhere in the OG implementation
    rng = jax.random.PRNGKey(42)
    rng, spawn_rng = jax.random.split(rng)
    # Update spawns
    spawn_state, new_shark_positions, new_sub_positions = update_enemy_spawns(
        spawn_state,
        new_shark_positions,
        new_sub_positions,
        state.step_counter,
        spawn_rng
    )

    # spawn new divers
    diver_positions = spawn_divers(spawn_state, state.diver_positions, new_shark_positions, new_sub_positions, state.step_counter)

    return spawn_state, new_shark_positions, new_sub_positions, diver_positions


def enemy_missiles_step(curr_sub_positions, curr_enemy_missile_positions, step_counter) -> chex.Array:
    def single_missile_step(i, carry):
        # Input i is the loop index, carry is the full array of missile positions
        # Get current submarine and missile for this index
        sub_pos = curr_sub_positions[i]
        missile_pos = carry[i]

        # check if the missile is in frame
        missile_exists = missile_pos[2] != 0

        # check if the missile should be spawned
        should_spawn = jnp.logical_and(
            ~missile_exists,
            jnp.logical_and(
                sub_pos[0] >= MISSILE_SPAWN_POSITIONS[0],
                sub_pos[0] <= MISSILE_SPAWN_POSITIONS[1]
            )
        )

        # Calculate new missile position
        new_missile = jnp.where(
            should_spawn,
            jnp.array([sub_pos[0] + 4, ENEMY_MISSILE_Y[i], sub_pos[2]]),  # Use submarine's direction
            missile_pos
        )

        # Move existing missile
        new_missile = jnp.where(
            missile_exists,
            jnp.array([new_missile[0] + new_missile[2], new_missile[1], new_missile[2]]),
            new_missile
        )

        # Check bounds
        new_missile = jnp.where(
            new_missile[0] < X_BORDERS[0],
            jnp.array([0, 0, 0]),
            jnp.where(
                new_missile[0] > X_BORDERS[1],
                jnp.array([0, 0, 0]),
                new_missile
            )
        )

        # Update the missile position in the full array
        return carry.at[i].set(new_missile)

    # Update all missile positions maintaining the array shape
    new_missile_positions = jax.lax.fori_loop(
        0, curr_sub_positions.shape[0],
        single_missile_step,
        curr_enemy_missile_positions
    )

    return new_missile_positions

def player_missile_step(state: State, curr_player_x, curr_player_y, action: chex.Array) -> chex.Array:
    # check if the player shot this frame
    fire = jnp.any(jnp.array([action == FIRE, action == UPRIGHTFIRE, action == UPLEFTFIRE, action == DOWNFIRE, action == DOWNRIGHTFIRE, action == DOWNLEFTFIRE, action == RIGHTFIRE, action == LEFTFIRE, action == UPFIRE]))

    # IMPORTANT: do not change the order of this check, since the missile does not move in its first frame!!
    # also check if there is currently a missile in frame by checking if the player_missile_position is empty
    missile_exists = state.player_missile_position[2] != 0

    # TODO: make it clearer that the missile has a 7 pixel y position offset
    # if the player shot and there is no missile in frame, then we can shoot a missile
    # the missile y is the current player y position + 7
    # the missile x is either player x + 3 if facing left or player x + 13 if facing right
    new_missile = jnp.where(
        jnp.logical_and(fire, ~missile_exists),
        jnp.where(
            state.player_direction == -1,
            jnp.array([curr_player_x + 3, curr_player_y + 7, -1]),
            jnp.array([curr_player_x + 13, curr_player_y + 7, 1])
        ),
        state.player_missile_position
    )

    # if a missile is in frame and exists, we move the missile further in the specified direction (5 per tick), also always put the missile at the current player y position
    new_missile = jnp.where(
        missile_exists,
        jnp.array([new_missile[0] + new_missile[2]*5, curr_player_y + 7, new_missile[2]]),
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


def update_oxygen(state, player_x, player_y, player_missile_position):
    """Update oxygen levels and handle surfacing mechanics with proper surfacing detection"""
    PLAYER_BREATHING_Y = [47, 52]  # Range where oxygen neither increases nor decreases

    # Detect actual surfacing moment
    at_surface = player_y == 46
    was_underwater = player_y > 46
    just_surfaced = jnp.logical_and(at_surface, state.just_surfaced == 0)

    # Check player state
    decrease_ox = player_y > PLAYER_BREATHING_Y[1]
    has_divers = state.divers_collected > 0
    has_all_divers = state.divers_collected >= 6
    needs_oxygen = state.oxygen < 64

    # Special handling for initialization state
    in_init_state = state.just_surfaced == -1
    started_diving = player_y > PLAYER_START_Y
    filling_init_oxygen = jnp.logical_and(in_init_state, state.oxygen < 64)  # New condition

    # Surfacing conditions
    increase_ox = jnp.logical_and(at_surface, needs_oxygen)
    stay_same = jnp.logical_and(player_y >= PLAYER_BREATHING_Y[0],
                               player_y <= PLAYER_BREATHING_Y[1])

    # Handle surfacing without divers - prevent during init
    lose_life = jnp.logical_and(
        jnp.logical_and(just_surfaced, ~has_divers),
        ~in_init_state
    )

    # Handle surfacing with all divers
    should_reset = jnp.logical_and(just_surfaced, has_all_divers)

    # Update divers when surfacing - prevent during init
    new_divers_collected = jnp.where(
        jnp.logical_and(just_surfaced, has_divers),
        jnp.where(
            in_init_state,
            state.divers_collected,  # Keep divers during init
            state.divers_collected - 1  # Normal surfacing loss
        ),
        state.divers_collected
    )

    # Update surfacing flag - only leave init when oxygen full AND diving starts
    new_just_surfaced = jnp.where(
        in_init_state,
        jnp.where(
            jnp.logical_and(started_diving, state.oxygen >= 64),  # Both conditions must be met
            jnp.array(0),   # Start normal underwater/surface cycle
            jnp.array(-1)   # Stay in init until conditions met
        ),
        jnp.where(
            was_underwater,
            jnp.array(0),   # Reset flag when going underwater
            jnp.where(
                at_surface,
                jnp.array(1),  # Set flag when at surface
                state.just_surfaced
            )
        )
    )

    # REPLACED: Oxygen handling with init state consideration
    new_oxygen = jnp.where(
        filling_init_oxygen,
        jnp.where(
            state.step_counter % 2 == 0,  # Fill every other frame during init
            state.oxygen + 1,
            state.oxygen
        ),
        jnp.where(
            decrease_ox,
            jnp.where(
                state.step_counter % 32 == 0,
                state.oxygen - 1,
                state.oxygen
            ),
            state.oxygen
        )
    )

    # Handle oxygen refill at surface (not during init)
    can_refill = jnp.logical_and(increase_ox, has_divers)
    new_oxygen = jnp.where(
        jnp.logical_and(can_refill, ~in_init_state),
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

    new_oxygen = jnp.where(
        stay_same,
        state.oxygen,
        new_oxygen
    )

    # Block player movement and disable firing while surfacing with divers
    player_x = jnp.where(
        can_refill,
        state.player_x,
        player_x
    )

    player_y = jnp.where(
        can_refill,
        46,  # Force to y=46 while refilling
        player_y
    )

    player_missile_position = jnp.where(
        can_refill,
        jnp.zeros(3),
        player_missile_position
    )

    # Prevent oxygen depletion during init
    oxygen_depleted = jnp.logical_and(
        new_oxygen <= jnp.array(0),
        ~in_init_state
    )

    return (new_oxygen, player_x, player_y, player_missile_position,
            oxygen_depleted, lose_life, new_divers_collected, should_reset, new_just_surfaced)

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

    # perform out of bounds checks
    player_x = jnp.where(
        player_x < PLAYER_BOUNDS[0][0],
        PLAYER_BOUNDS[0][0],  # Clamp to min player bound
        jnp.where(
            player_x > PLAYER_BOUNDS[0][1],
            PLAYER_BOUNDS[0][1],  # Clamp to max player bound
            player_x
        )
    )

    player_y = jnp.where(
        player_y < PLAYER_BOUNDS[1][0],
        PLAYER_BOUNDS[1][0],
        jnp.where(
            player_y > PLAYER_BOUNDS[1][1],
            PLAYER_BOUNDS[1][1],
            player_y
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
            diver_positions=jnp.zeros((MAX_DIVERS, 3)),  # 4 divers
            shark_positions=jnp.zeros((MAX_SHARKS, 3)),
            sub_positions=jnp.zeros((MAX_SUBS, 3)),  # x, y, direction
            enemy_missile_positions=jnp.zeros((MAX_ENEMY_MISSILES, 3)),  # 4 missiles
            surface_sub_position=jnp.zeros((MAX_SURFACE_SUBS, 3)),  # 1 surface sub
            player_missile_position=jnp.zeros(3),  # x,y,direction
            step_counter=jnp.array(0),
            just_surfaced=jnp.array(-1)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        # First check if player should be frozen for oxygen refill
        needs_oxygen = jnp.logical_and(state.player_y == 46, state.oxygen < 64)

        # Calculate potential movement
        next_x, next_y, next_direction = player_step(state, action)

        # Override movement if refilling oxygen
        player_x = jnp.where(needs_oxygen, state.player_x, next_x)
        player_y = jnp.where(needs_oxygen, 46, next_y)
        player_direction = jnp.where(needs_oxygen, state.player_direction, next_direction)

        # Process missile only if not refilling oxygen
        player_missile_position = jnp.where(
            needs_oxygen,
            jnp.zeros(3),  # No missile when refilling
            player_missile_step(state, player_x, player_y, action)
        )

        # Update oxygen and handle surfacing mechanics
        new_oxygen, player_x, player_y, player_missile_position, oxygen_depleted, lose_life_surfacing, new_divers_collected, should_reset, new_just_surfaced = update_oxygen(
            state, player_x, player_y, player_missile_position
        )

        # Update divers collected count from oxygen mechanics
        state = state._replace(divers_collected=new_divers_collected)

        # Check missile collisions
        player_missile_position, new_shark_positions, new_sub_positions, new_score = check_missile_collisions(
            player_missile_position,
            state.shark_positions,
            state.sub_positions,
            state.score
        )

        # perform all necessary spawn steps
        new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions = spawn_step(
            state,
            state.spawn_state,
            new_shark_positions,
            new_sub_positions,
            state.diver_positions,
        )

        new_diver_positions, new_divers_collected = step_diver_movement(
            new_diver_positions,
            new_shark_positions,
            player_x,
            player_y,
            state.divers_collected,
            state.step_counter
        )

        # update the enemy missile positions
        new_enemy_missile_positions = enemy_missiles_step(
            new_sub_positions,
            state.enemy_missile_positions,
            state.step_counter
        )

        # check if the player has collided with any of the enemies
        player_collision = check_player_collision(player_x, player_y, new_sub_positions, new_shark_positions, state.enemy_missile_positions)

        # perform all live loosing checks TODO: add other checks as needed
        lose_life = jnp.any(jnp.array([oxygen_depleted, player_collision, lose_life_surfacing]))

        # TODO: insert the timer for the death animation
        # if the player has lost a life, reset everything except lives, score and collected divers
        reset_state = self.reset()._replace(
            lives=state.lives - 1,
            score=state.score,
            divers_collected=state.divers_collected - 1
        )

        # State for scoring all divers
        BONUS_POINTS = 1000  # 1000 points bonus in the original game
        scoring_state = self.reset()._replace(
            lives=state.lives,
            score=state.score + BONUS_POINTS
        )

        # Create the normal returned state
        normal_returned_state = State(
            player_x=player_x,
            player_y=player_y,
            player_direction=player_direction,
            oxygen=new_oxygen,
            divers_collected=new_divers_collected,
            score=new_score,
            lives=state.lives,
            spawn_state=new_spawn_state,
            diver_positions=new_diver_positions,
            shark_positions=new_shark_positions,
            sub_positions=new_sub_positions,
            enemy_missile_positions=new_enemy_missile_positions,
            surface_sub_position=jnp.zeros((MAX_SURFACE_SUBS, 3)),
            player_missile_position=player_missile_position,
            step_counter=state.step_counter + 1,
            just_surfaced=new_just_surfaced
        )

        # First handle surfacing with all divers (scoring)
        intermediate_state = jax.lax.cond(
            should_reset,  # Condition from update_oxygen when all divers collected
            lambda _: scoring_state,
            lambda _: normal_returned_state,
            operand=None
        )

        # Then handle life loss
        final_state = jax.lax.cond(
            lose_life,
            lambda _: reset_state,
            lambda _: intermediate_state,
            operand=None
        )

        # Return unchanged state for now
        return final_state

class Renderer_AtraJaxis:
    def __init__(self):
        self.sprite_bg = SPRITE_BG
        self.sprite_pl_sub = SPRITE_PL_SUB
        self.sprite_diver = SPRITE_DIVER

    #     self.spriteLoader.loadFrame('sprites\seaquest\player_sub\\1.npy', name='pl_sub1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\player_sub\\2.npy', name='pl_sub2')
    #     self.spriteLoader.loadFrame('sprites\seaquest\player_sub\\3.npy', name='pl_sub3')
    #     self.spriteLoader.loadSprite('player_sub', [('pl_sub1', 4), ('pl_sub2', 4), ('pl_sub3', 4)], RenderMode.LOOP)
        
    #     # enemy submarine
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_sub\\1.npy', name='en_sub1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_sub\\2.npy', name='en_sub2')
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_sub\\3.npy', name='en_sub3')
    #     self.spriteLoader.loadSprite('enemy_sub', [('en_sub1', 4), ('en_sub2', 4), ('en_sub3', 4)], RenderMode.LOOP)
        
    #     # enemy shark
    #     self.spriteLoader.loadFrame('sprites\seaquest\shark\\1.npy', name='shark1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\shark\\2.npy', name='shark2')
    #     self.spriteLoader.loadSprite('shark', [('shark1', 16), ('shark2', 8)], RenderMode.LOOP)
        
    #     # diver
    #     self.spriteLoader.loadFrame('sprites\seaquest\diver\\1.npy', name='diver1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\diver\\2.npy', name='diver2')
    #     self.spriteLoader.loadSprite('diver', [('diver1', 16), ('diver2', 8)], RenderMode.LOOP)
        
    #     # player torpedo
    #     self.spriteLoader.loadFrame('sprites\seaquest\player_torp\\1.npy', name='pl_torpedo1')
    #     self.spriteLoader.loadSprite('player_torpedo', [('pl_torpedo1', 1)], RenderMode.LOOP)
        
    #     # enemy torpedo
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_torp\\1.npy', name='en_torpedo1')
    #     self.spriteLoader.loadSprite('enemy_torpedo', [('en_torpedo1', 1)], RenderMode.LOOP)
        
        # initialize renderer

        
        

    # def __init__(self):
    #     # initialize renderer
    #     self.window_width = 160
    #     self.window_height = 210
    #     self.scaling_factor = 3
    #     pygame.init()
    #     self.win = pygame.display.set_mode((self.window_width*self.scaling_factor, self.window_height*self.scaling_factor))
    #     pygame.display.set_caption("Seaquest")
    #     self.clock = pygame.time.Clock()
    #     self.font = pygame.font.Font(None, 36)
    #     running = True
        
    #     # initialize sprites
    #     self.spriteLoader = SpriteLoader()
        
    #     # background
    #     self.spriteLoader.loadFrame('sprites\seaquest\\bg\\1.npy', name='bg1')
    #     self.spriteLoader.loadSprite('bg', [('bg1', 1)], RenderMode.LOOP)
        
    #     # player submarine
    #     self.spriteLoader.loadFrame('sprites\seaquest\player_sub\\1.npy', name='pl_sub1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\player_sub\\2.npy', name='pl_sub2')
    #     self.spriteLoader.loadFrame('sprites\seaquest\player_sub\\3.npy', name='pl_sub3')
    #     self.spriteLoader.loadSprite('player_sub', [('pl_sub1', 4), ('pl_sub2', 4), ('pl_sub3', 4)], RenderMode.LOOP)
        
    #     # enemy submarine
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_sub\\1.npy', name='en_sub1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_sub\\2.npy', name='en_sub2')
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_sub\\3.npy', name='en_sub3')
    #     self.spriteLoader.loadSprite('enemy_sub', [('en_sub1', 4), ('en_sub2', 4), ('en_sub3', 4)], RenderMode.LOOP)
        
    #     # enemy shark
    #     self.spriteLoader.loadFrame('sprites\seaquest\shark\\1.npy', name='shark1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\shark\\2.npy', name='shark2')
    #     self.spriteLoader.loadSprite('shark', [('shark1', 16), ('shark2', 8)], RenderMode.LOOP)
        
    #     # diver
    #     self.spriteLoader.loadFrame('sprites\seaquest\diver\\1.npy', name='diver1')
    #     self.spriteLoader.loadFrame('sprites\seaquest\diver\\2.npy', name='diver2')
    #     self.spriteLoader.loadSprite('diver', [('diver1', 16), ('diver2', 8)], RenderMode.LOOP)
        
    #     # player torpedo
    #     self.spriteLoader.loadFrame('sprites\seaquest\player_torp\\1.npy', name='pl_torpedo1')
    #     self.spriteLoader.loadSprite('player_torpedo', [('pl_torpedo1', 1)], RenderMode.LOOP)
        
    #     # enemy torpedo
    #     self.spriteLoader.loadFrame('sprites\seaquest\enemy_torp\\1.npy', name='en_torpedo1')
    #     self.spriteLoader.loadSprite('enemy_torpedo', [('en_torpedo1', 1)], RenderMode.LOOP)
        

        
    #     # digits
    #     char_to_frame = {}
    #     for i in range(10):
    #         self.spriteLoader.loadFrame(f'sprites\seaquest\digits\\{i}.npy', name=f'digit{i}')
    #         char_to_frame[str(i)] = self.spriteLoader.frames[f'digit{i}']
            

            
    #     # life indicator
    #     self.spriteLoader.loadFrame('sprites\seaquest\life_indicator\\1.npy', name='life1')
    #     char_to_frame['l'] = self.spriteLoader.frames['life1']

        
    #     # diver indicator
    #     self.spriteLoader.loadFrame('sprites\seaquest\diver_indicator\\1.npy', name='diver_indicator1')
    #     char_to_frame['d'] = self.spriteLoader.frames['diver_indicator1']

    #     # initialize canvas
    #     self.canvas = Canvas(self.window_width, self.window_height)
    #     self.canvas.addLayer(Layer('bg', self.window_width, self.window_height))
    #     self.canvas.addLayer(Layer('torpedoes', self.window_width, self.window_height))
    #     self.canvas.addLayer(Layer('player_sub', self.window_width, self.window_height))
    #     self.canvas.addLayer(Layer('divers', self.window_width, self.window_height))
    #     self.canvas.addLayer(Layer('enemies', self.window_width, self.window_height))
    #     self.canvas.addLayer(Layer('waves', self.window_width, self.window_height))
    #     self.canvas.addLayer(Layer('HUD', self.window_width, self.window_height))
        
    #     # initialize game objects
    #     background = GameObject(0, 0, self.spriteLoader.getSprite('bg'))
    #     self.canvas.getLayer('bg').addGameObject(background)
        
    #     pl_sub = GameObject(0, 0, self.spriteLoader.getSprite('player_sub'))
    #     self.canvas.getLayer('player_sub').addGameObject(pl_sub)
        
    #     # HUD elements
    #     # TODO: verify positioning and width between chars in the HUD

    #     hud_score = TextHUD("00", 10, 10, char_to_frame, 2) # score indicator
    #     self.hud_score = hud_score
    #     self.canvas.getLayer('HUD').addGameObject(hud_score)
        
    #     hud_lives = TextHUD("lll", 20, 10, char_to_frame, 2) # lives indicator
    #     self.hud_lives = hud_lives
    #     self.canvas.getLayer('HUD').addGameObject(hud_lives)
        
    #     hud_divers = TextHUD("", 20, 40, char_to_frame, 2) # diver indicator
    #     self.hud_divers = hud_divers
    #     self.canvas.getLayer('HUD').addGameObject(hud_divers)

    #     hud_oxygen = BarHUD(10, 60, 60, 5, 64, 64, (255,255,255,255)) # oxygen bar
    #     self.hud_oxygen = hud_oxygen
    #     self.canvas.getLayer('HUD').addGameObject(hud_oxygen)

    #     # initialize arrays that map indices to game objects
    #     self.diver_objects = [None] * MAX_DIVERS
    #     self.shark_objects = [None] * MAX_SHARKS
    #     self.sub_objects = [None] * MAX_SUBS
    #     self.enemy_torpedo_objects = [None] * MAX_ENEMY_MISSILES
    #     self.surface_sub_objects = [None] * MAX_SURFACE_SUBS
    #     self.player_torpedo_object = None

    @partial(jax.jit, static_argnums=(0,))
    def render(self, state):
        raster = jnp.zeros((WIDTH, HEIGHT, 3))

        # render background
        frame_bg = aj.get_sprite_frame(self.sprite_bg, 0)
        raster = aj.render_at(raster, 0, 0, frame_bg, flip_vertical=True)

        # render player submarine
        frame_pl_sub = aj.get_sprite_frame(self.sprite_pl_sub, state.step_counter)
        raster = aj.render_at(raster, state.player_y, state.player_x, frame_pl_sub, flip_vertical = state.player_direction == FACE_LEFT)
        # render divers
        frame_diver = aj.get_sprite_frame(self.sprite_diver, state.step_counter)
        diver_positions = state.diver_positions

        # convert diver positions to integers
        def render_diver(i, raster_carry):
            should_render = diver_positions[i][0] > 0
            return jax.lax.cond(
                should_render,
                lambda r: aj.render_at(
                    r,
                    diver_positions[i][1],
                    diver_positions[i][0],
                    frame_diver,
                    flip_horizontal=(diver_positions[i][2] == FACE_LEFT)
                ),
                lambda r: r,
                raster_carry
            )

        # Use fori_loop to render all divers
        raster_rendered_diver = jax.lax.fori_loop(
            0,
            MAX_DIVERS,
            render_diver,
            raster
        )

        return raster_rendered_diver


    #     # update divers
    #     for idx in range(MAX_DIVERS):
    #         if state.diver_positions[idx][0] > 0: # indicates existence
    #             diver_x = int(state.diver_positions[idx][1].item())
    #             diver_y = int(state.diver_positions[idx][0].item())
    #             diver_direction = state.diver_positions[idx][2].item()
    #             if self.diver_objects[idx] is None: # if object does not exist, create it
    #                 self.diver_objects[idx] = GameObject(diver_x, diver_y, self.spriteLoader.getSprite('diver'))
    #                 self.canvas.getLayer('divers').addGameObject(self.diver_objects[idx])
    #                 self.canvas.getLayer('divers').gameObjects[idx].sprite.transform["flip_horizontal"] = diver_direction == FACE_LEFT # flip sprite if facing left

    #             else: # if object exists, update its position
    #                 self.diver_objects[idx].displace(diver_x, diver_y)
    #                 if self.diver_objects[idx] is not None: # avoid error if diver object is killed on the same frame
    #                     self.diver_objects[idx].sprite.transform["flip_horizontal"] = diver_direction == FACE_LEFT # flip sprite if facing left

    #         else: # the diver no longer exists
    #             if self.diver_objects[idx] is not None:
    #                 self.canvas.getLayer('divers').removeGameObject(self.diver_objects[idx])
    #                 self.diver_objects[idx] = None
                    
    #     # update sharks
    #     for idx in range(MAX_SHARKS):
    #         if state.shark_positions[idx][0] > 0: # indicates existence
    #             shark_x = int(state.shark_positions[idx][1].item())
    #             shark_y = int(state.shark_positions[idx][0].item())
    #             shark_direction = state.shark_positions[idx][2].item()
    #             if self.shark_objects[idx] is None: # if object does not exist, create it
    #                 self.shark_objects[idx] = GameObject(shark_x, shark_y, self.spriteLoader.getSprite('shark'))
    #                 self.canvas.getLayer('enemies').addGameObject(self.shark_objects[idx])
    #                 self.shark_objects[idx].sprite.transform["flip_horizontal"] = shark_direction == FACE_LEFT # flip sprite if facing left

    #             else: # if object exists, update its position
    #                 self.shark_objects[idx].displace(shark_x, shark_y)
    #                 if self.shark_objects[idx] is not None: # avoid error if shark object is killed on the same frame
    #                     self.shark_objects[idx].sprite.transform["flip_horizontal"] = shark_direction == FACE_LEFT # flip sprite if facing left

    #         else: # the shark no longer exists
    #             if self.shark_objects[idx] is not None:
    #                 self.canvas.getLayer('enemies').removeGameObject(self.shark_objects[idx])
    #                 self.shark_objects[idx] = None
                    
    #     # update enemy submarines
    #     for idx in range(MAX_SUBS):
    #         if state.sub_positions[idx][0] > 0: # indicates existence
    #             sub_x = int(state.sub_positions[idx][1].item())
    #             sub_y = int(state.sub_positions[idx][0].item())
    #             sub_direction = state.sub_positions[idx][2].item()
    #             if self.sub_objects[idx] is None:
    #                 self.sub_objects[idx] = GameObject(sub_x, sub_y, self.spriteLoader.getSprite('enemy_sub'))
    #                 self.canvas.getLayer('enemies').addGameObject(self.sub_objects[idx])
    #                 # update direction of submarine
    #                 self.sub_objects[idx].sprite.transform["flip_horizontal"] = sub_direction == FACE_LEFT
    #             else:
    #                 self.sub_objects[idx].displace(sub_x, sub_y)
    #                 if self.sub_objects[idx] is not None: # avoid error if submarine object is killed on the same frame
    #                     self.sub_objects[idx].sprite.transform["flip_horizontal"] = sub_direction == FACE_LEFT
    #         else: # the submarine no longer exists
    #             if self.sub_objects[idx] is not None:
    #                 self.canvas.getLayer('enemies').removeGameObject(self.sub_objects[idx])
    #                 self.sub_objects[idx] = None
                    
    #     # update surface submarine
    #     for idx in range(MAX_SURFACE_SUBS):
    #         if state.surface_sub_position[idx][0] > 0: # indicates existence
    #             surface_sub_x = int(state.surface_sub_position[idx][1].item())
    #             surface_sub_y = int(state.surface_sub_position[idx][0].item())
    #             sub_direction = state.surface_sub_position[idx][2].item()
    #             if self.surface_sub_objects[idx] is None:
    #                 self.surface_sub_objects[idx] = GameObject(surface_sub_x, surface_sub_y, self.spriteLoader.getSprite('enemy_sub'))
    #                 self.canvas.getLayer('enemies').addGameObject(self.surface_sub_objects[idx])
    #                 # update direction of submarine
    #                 self.surface_sub_objects[idx].sprite.transform["flip_horizontal"] = sub_direction == FACE_LEFT
    #             else:
    #                 # if object exists, update its position
    #                 self.surface_sub_objects[idx].displace(surface_sub_x, surface_sub_y)
    #                 # update direction of submarine
    #                 self.surface_sub_objects[idx].sprite.transform["flip_horizontal"] = sub_direction == FACE_LEFT
            
                    
    #     # update player's torpedo
    #     if state.player_missile_position[0] > 0: # exists
    #         pltorp_x = int(state.player_missile_position[1].item())
    #         pltorp_y = int(state.player_missile_position[0].item())
    #         if self.player_torpedo_object is None: # if object does not exist, create it
    #             self.player_torpedo_object = GameObject(pltorp_x, pltorp_y, self.spriteLoader.getSprite('player_torpedo'))
    #             self.canvas.getLayer('torpedoes').addGameObject(self.player_torpedo_object)
    #         else: # if object exists, update its position
    #             self.player_torpedo_object.displace(pltorp_x, pltorp_y)   
    #     else: # the torpedo no longer exists
    #         if self.player_torpedo_object is not None:
    #             self.canvas.getLayer('torpedoes').removeGameObject(self.player_torpedo_object)
    #             self.player_torpedo_object = None
                
    #     # update enemy torpedoes
    #     for idx in range(MAX_ENEMY_MISSILES):
    #         if state.enemy_missile_positions[idx][0] > 0: # indicates existence
    #             entorp_x = int(state.enemy_missile_positions[idx][1].item())
    #             entorp_y = int(state.enemy_missile_positions[idx][0].item())
    #             if self.enemy_torpedo_objects[idx] is None: # if object does not exist, create it
    #                 self.enemy_torpedo_objects[idx] = GameObject(entorp_x, entorp_y, self.spriteLoader.getSprite('enemy_torpedo'))
    #                 self.canvas.getLayer('torpedoes').addGameObject(self.enemy_torpedo_objects[idx])
    #             else: # if object exists, update its position
    #                 self.enemy_torpedo_objects[idx].displace(entorp_x, entorp_y)
    #         else: # the torpedo no longer exists
    #             if self.enemy_torpedo_objects[idx] is not None:
    #                 self.canvas.getLayer('torpedoes').removeGameObject(self.enemy_torpedo_objects[idx])
    #                 self.enemy_torpedo_objects[idx] = None
        
    #     # update HUD elements
    #     self.hud_score.text = str(int(state.score.item())) 
    #     self.hud_lives.text = "l" * int(state.lives.item()) 
    #     self.hud_divers.text = "d" * int(state.divers_collected.item() % 7) 
    #     self.hud_oxygen.current_value = state.oxygen.item()
    #     # finally, update the canvas
    #     self.canvas.update()




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
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()
   
    
    renderer_AtraJaxis = Renderer_AtraJaxis()

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

        # render and update pygame
        raster = renderer_AtraJaxis.render(curr_state)
        aj.update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        # renderer.render(curr_state)
        counter += 1
        clock.tick(60)
        # renderer.clock.tick(256)

    pygame.quit()