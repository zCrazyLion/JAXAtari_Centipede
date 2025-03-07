import sys
from functools import partial
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
import chex
import pygame


# TODO: is it implemented that you get a submarine at 10000 points?
# TODO: surface submarine at 6 divers collected + difficulty 1
# Game Constants
WINDOW_WIDTH = 160 * 3
WINDOW_HEIGHT = 210 * 3

# Colors
BACKGROUND_COLOR = (0, 0, 139)  # Dark blue for water
PLAYER_COLOR = (187, 187, 53)  # Yellow for player sub
DIVER_COLOR = (66, 72, 200)  # Pink for divers
SHARK_DIFFICULTY_COLORS = jnp.array([
    [92, 186, 92],   # Level 0: Base green
    [213, 130, 74],  # Level 1: Orange (adjusted from original ROM)
    [170, 92, 170],  # Level 2: Purple (adjusted from original ROM COLOR_KILLER_SHARK_02)
    [213, 92, 130],  # Level 3: Pink (adjusted from original ROM)
    [186, 92, 92],   # Level 4: Red (adjusted from original ROM)
])
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

SPAWN_POSITIONS_Y = jnp.array([71, 95, 119, 139]) # submarines at y=69?
SUBMARINE_Y_OFFSET = 2
ENEMY_MISSILE_Y = jnp.array([73, 97, 121, 141]) # missile x = submarine.x + 4
DIVER_SPAWN_POSITIONS = jnp.array([69, 93, 117, 141])

MISSILE_SPAWN_POSITIONS = jnp.array([39, 126])  # Right, Left

# First wave directions from original code
FIRST_WAVE_DIRS = jnp.array([False, False, False, True])

class SpawnState(NamedTuple):
    difficulty: chex.Array  # Current difficulty level (0-7)
    lane_dependent_pattern: chex.Array  # Track waves independently per lane [4 lanes]
    to_be_spawned: chex.Array  # tracks which enemies are still in the spawning cycle [4 lanes * 3 slots] -> necessary due to the spaced out spawning of multiple enemies
    survived: chex.Array # track if last enemy survived [4 lanes * 3 slots]
    prev_sub: chex.Array  # Track previous entity type for each lane [4 lanes]
    spawn_timers: chex.Array  # Individual spawn timers per lane [4 lanes]
    diver_array: chex.Array # Track which divers are still in the spawning cycle [4 lanes]

def initialize_spawn_state() -> SpawnState:
    """Initialize spawn state with first wave matching original game."""
    return SpawnState(
        difficulty=jnp.array(0),
        lane_dependent_pattern=jnp.zeros(4, dtype=jnp.int32),  # Each lane starts at wave 0
        to_be_spawned=jnp.zeros(12, dtype=jnp.int32),  # Track which enemies are still in the spawning cycle
        survived=jnp.zeros(12, dtype=jnp.bool_),  # Track which enemies survived
        prev_sub=jnp.ones(4, dtype=jnp.int32),  # Track previous entity type (0 if shark, 1 if sub) -> starts at 1 since the first wave is sharks
        spawn_timers=jnp.array([277, 277, 277, 277 + 60], dtype=jnp.int32),  # 277 is the std starting timer in the base game
        diver_array=jnp.zeros(4, dtype=jnp.int32),
    )


def soft_reset_spawn_state(spawn_state: SpawnState) -> SpawnState:
    """ Reset spawn_times"""
    return spawn_state._replace(
        spawn_timers=jnp.array([277, 277, 277, 277 + 60], dtype=jnp.int32)
    )

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
    shark_positions: chex.Array  # (12, 3) array for sharks - separated into 4 lanes, 3 slots per lane [left to right]
    sub_positions: chex.Array  # (12, 3) array for enemy subs - separated into 4 lanes, 3 slots per lane [left to right]
    enemy_missile_positions: chex.Array  # (4, 3) array for enemy missiles (only the front boats can shoot)
    surface_sub_position: chex.Array  # (1, 3) array for surface submarine
    player_missile_position: chex.Array  # (1, 3) array for player missile (x, y, direction)
    step_counter: chex.Array
    just_surfaced: chex.Array  # Flag for tracking actual surfacing moment
    successful_rescues: chex.Array # Number of times the player has surfaced with all six divers
    death_counter: chex.Array  # Counter for tracking death animation


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
        score: chex.Array,
        successful_rescues: chex.Array,
        spawn_state: SpawnState
) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, SpawnState]:
    """Check for collisions between player missile and enemies

    Args:
        missile_pos: Player missile position
        shark_positions: Shark positions array
        sub_positions: Submarine positions array
        score: Current score
        successful_rescues: Number of successful rescue missions
        spawn_state: Current spawn state

    Returns:
        Tuple of:
        - Updated missile position
        - Updated shark positions
        - Updated sub positions
        - Updated score
        - Updated spawn state
    """
    # Create missile position array for collision check
    missile_rect_pos = jnp.array([missile_pos[0], missile_pos[1]])
    missile_active = missile_pos[2] != 0

    def check_enemy_collisions(enemy_idx, carry):
        # Unpack carry state using named tuple
        missile_pos, shark_positions, sub_positions, score, spawn_state = carry

        # Check shark collisions - only if missile is active
        shark_collision = jnp.logical_and(
            missile_active,
            check_collision(
                missile_rect_pos, MISSILE_SIZE,
                shark_positions[enemy_idx], SHARK_SIZE
            )
        )

        # Check submarine collisions - only if missile is active
        sub_collision = jnp.logical_and(
            missile_active,
            check_collision(
                missile_rect_pos, MISSILE_SIZE,
                sub_positions[enemy_idx], ENEMY_SUB_SIZE
            )
        )

        # Update positions and score - use where instead of if statements
        new_shark_pos = jnp.where(
            shark_collision,
            jnp.zeros_like(shark_positions[enemy_idx]),
            shark_positions[enemy_idx]
        )

        new_sub_pos = jnp.where(
            sub_collision,
            jnp.zeros_like(sub_positions[enemy_idx]),
            sub_positions[enemy_idx]
        )

        # Update score
        score_increase = jnp.where(
            shark_collision,
            calculate_kill_points(successful_rescues),
            jnp.where(sub_collision, calculate_kill_points(successful_rescues), 0)
        )

        # Remove missile if it hit anything
        new_missile_pos = jnp.where(
            jnp.logical_or(shark_collision, sub_collision),
            jnp.array([0., 0., 0.]),
            missile_pos
        )

        # Update the kill tracking in spawn state
        any_collision = jnp.logical_or(shark_collision, sub_collision)
        new_survived = spawn_state.survived.at[enemy_idx].set(
            jnp.where(any_collision,
                      False,  # Set to False when destroyed by missile
                      spawn_state.survived[enemy_idx])
        )

        # Update spawn timers when enemy destroyed
        new_spawn_timers = spawn_state.spawn_timers.at[(enemy_idx // 3).astype(int)].set(  # Divide by 3 to get lane index
            jnp.where(any_collision, 60, spawn_state.spawn_timers[(enemy_idx // 3).astype(int)].astype(int))
        )

        # Create updated spawn state
        new_spawn_state = SpawnState(
            difficulty=spawn_state.difficulty,
            lane_dependent_pattern=spawn_state.lane_dependent_pattern,  # Wave counts updated in spawn step
            to_be_spawned=spawn_state.to_be_spawned,  # No changes to spawn state
            spawn_timers=new_spawn_timers,
            prev_sub=spawn_state.prev_sub,
            survived=new_survived,
            diver_array=spawn_state.diver_array
        )

        return (new_missile_pos,
                shark_positions.at[enemy_idx].set(new_shark_pos),
                sub_positions.at[enemy_idx].set(new_sub_pos),
                score + score_increase,
                new_spawn_state)

    # Initialize carry state
    init_carry = (missile_pos, shark_positions, sub_positions, score, spawn_state)

    # Always run the loop, but collisions only happen if missile is active
    return jax.lax.fori_loop(
        0, shark_positions.shape[0],
        check_enemy_collisions,
        init_carry
    )


def check_player_collision(player_x, player_y, submarine_list, shark_list, surface_sub_pos, enemy_projectile_list, score, successful_rescues) ->  Tuple[chex.Array, chex.Array]:
    # check if the player has collided with any of the three given lists
    # the player is a 16x11 rectangle
    # the submarine is a 8x11 rectangle
    # the shark is a 8x7 rectangle
    # the missile is a 8x1 rectangle
    # the surface submarine is 8x11 as well

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

    # check if the player collided with the surface submarine
    surface_collision = jnp.any(
        check_collision(
            jnp.array([player_x, player_y]),
            PLAYER_SIZE,
            jnp.array([surface_sub_pos]),
            ENEMY_SUB_SIZE
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

    # Calculate points for collisions.
    # When colliding with a shark or submarine the player gains points similar to killing the object
    collision_points = jnp.where(
        shark_collisions,
        calculate_kill_points(successful_rescues),
        jnp.where(
            submarine_collisions,
            calculate_kill_points(successful_rescues),
            jnp.where(
                surface_collision,
                calculate_kill_points(successful_rescues),
                0
            ))
    )

    return jnp.any(jnp.array([submarine_collisions, shark_collisions, missile_collisions, surface_collision])), collision_points


def get_spawn_position(moving_left: chex.Array, slot: chex.Array) -> chex.Array:
    """Get spawn position based on movement direction and slot number"""
    base_y = jnp.array(SPAWN_POSITIONS_Y[slot])
    x_pos = jnp.where(moving_left,
                      jnp.array(165, dtype=jnp.int32),  # Start right if moving left
                      jnp.array(0, dtype=jnp.int32))    # Start left if moving right
    direction = jnp.where(moving_left, -1, 1)  # -1 for left, 1 for right
    return jnp.array([x_pos, base_y, direction], dtype=jnp.int32)


def is_slot_empty(pos: chex.Array) -> chex.Array:
    """Check if a position slot is empty (0,0,ß)"""
    return pos[2] == 0


def get_front_entity(i, lane_positions):
    # check on the first submarine in the lane which direction they are going
    direction = lane_positions[0][2]

    direction = jnp.where(
        lane_positions[0][2] == 0,
        jnp.where(
            lane_positions[1][2] == 0,
            lane_positions[2][2],
            lane_positions[1][2]
        ),
        lane_positions[0][2]
    )

    # if direction is 1, go from right to left until an active entity is found
    # if direction is -1, go from left to right until an active entity is found
    front_entity = jnp.where(
        direction == -1,
        jnp.where(
            lane_positions[0][2] != 0,
            lane_positions[0],
            jnp.where(
                lane_positions[1][2] != 0,
                lane_positions[1],
                jnp.where(
                    lane_positions[2][2] != 0,
                    lane_positions[2],
                    jnp.zeros(3)
                )
            )
        ),
        jnp.where(
            lane_positions[2][2] != 0,
            lane_positions[2],
            jnp.where(
                lane_positions[1][2] != 0,
                lane_positions[1],
                jnp.where(
                    lane_positions[0][2] != 0,
                    lane_positions[0],
                    jnp.zeros(3)
                )
            )
        )
    )

    return front_entity


def get_pattern_for_difficulty(current_pattern: chex.Array, moving_left: chex.Array) -> chex.Array:
    """Returns spawn pattern based on the lane's current wave/pattern number

    Pattern meanings:
    0: Single enemy (initial pattern)
    1: Two adjacent enemies
    2: Two enemies with gap
    3: Three enemies in a row
    """
    # Basic pattern arrays for different formations
    PATTERNS = jnp.array([
        [0, 0, 1],  # wave 0: Single enemy
        [0, 1, 1],  # wave 1: Two adjacent
        [1, 0, 1],  # wave 2: Two with gap
        [1, 1, 1],  # wave 3: Three in row
    ])

    # Reverse pattern if moving left
    base_pattern = PATTERNS[current_pattern]
    pattern = jnp.where(
        moving_left,
        jnp.flip(base_pattern),  # Reverse pattern for left movement
        base_pattern
    )

    return pattern


def update_enemy_spawns(spawn_state: SpawnState,
                        shark_positions: chex.Array,
                        sub_positions: chex.Array,
                        step_counter: chex.Array) -> Tuple[SpawnState, chex.Array, chex.Array]:
    """Update enemy spawns using pattern-based system matching original game."""

    # TODO: inital pattern wrong
    def initialize_new_spawn_cycle(i, carry):
        spawn_state, shark_positions, sub_positions = carry

        # Get survived status for this lane (3 slots)
        lane_survived = jax.lax.dynamic_slice(
            spawn_state.survived,
            (i * 3,),
            (3,)
        )

        # Update the difficulty patterns for this lane
        left_over = jnp.any(lane_survived)
        clipped_difficulty = spawn_state.difficulty % 8
        # Update spawn state
        lane_specific_pattern = jnp.where(
            jnp.logical_not(left_over),  # Only update if all destroyed
            jnp.where(
                clipped_difficulty < 2, 0,
                jnp.where(
                    clipped_difficulty < 4, 1,
                    jnp.where(
                        clipped_difficulty < 6, 2,
                        jnp.where(
                            clipped_difficulty < 8, 3,
                            0
                        )
                    )
                )
            ),
            spawn_state.lane_dependent_pattern[i]
        )


        # find out the direction # TODO: this is wrong, fix
        moving_left = jnp.where(
            lane_specific_pattern == 0,
            FIRST_WAVE_DIRS[lane_specific_pattern],
            lane_specific_pattern % 2 == 1  # After first wave, alternate direction each wave
        )

        #print("left-over", left_over)
        #print("survived_all", spawn_state.survived)
        #print("lane_survived", lane_survived)

        # get the spawn pattern for this lane
        # Check if this slot had something survive last time (if yes, we have to overwrite the current_pattern)
        current_pattern = jnp.where(
            left_over,
            lane_survived,
            get_pattern_for_difficulty(lane_specific_pattern, moving_left)
        )
        #print("curr_pattern", current_pattern)

        # check if this should be a submarine or a shark
        is_sub = jnp.logical_and(
            left_over,
            jnp.logical_not(spawn_state.prev_sub[i])
        )

        # set the positions for the first enemy in the wave (dependent on the direction this is either the first or the last slot)
        first_slot = jnp.where(moving_left, 0, 2)

        base_pos = get_spawn_position(moving_left, jnp.array(i))
        # spawn the first enemy in the wave
        new_shark_positions = jnp.where(
            is_sub,
            shark_positions,
            shark_positions.at[(i * 3 + first_slot)].set(base_pos)
        )

        new_sub_positions = jnp.where(
            is_sub,
            sub_positions.at[(i * 3 + first_slot)].set(base_pos),
            sub_positions
        )

        # for the other two slots, set the corresponding positions in the to_be_spawned array to the current pattern (1 for right, -1 for left, 0 for no spawn scheduled)
        # -> helps us keep track of direction in the next step
        new_to_be_spawned = jnp.where(
            moving_left,
            -current_pattern,
            current_pattern
        )

        # wipe the survived status for this lane (since we are starting a new wave)
        indices = (i * 3 + jnp.array([0, 1, 2]))
        new_survived_full = spawn_state.survived.at[indices].set(jnp.zeros(3, dtype=jnp.bool_))

        # remove the spawned entity from the new_to_be_spawned array
        new_to_be_spawned = new_to_be_spawned.at[jnp.where(moving_left, 0, 2)].set(0)

        # Update the full to_be_spawned array for this lane
        new_full_to_be_spawned = spawn_state.to_be_spawned.at[indices].set(new_to_be_spawned)

        #print("full_to_be_spawned_after_init", new_full_to_be_spawned)

        new_spawn_state = SpawnState(
            difficulty=spawn_state.difficulty,
            lane_dependent_pattern=spawn_state.lane_dependent_pattern.at[i].set(lane_specific_pattern),
            to_be_spawned=new_full_to_be_spawned,
            survived=new_survived_full,
            prev_sub=spawn_state.prev_sub.at[i].set(is_sub),
            spawn_timers=spawn_state.spawn_timers.at[i].set(60),
            diver_array=spawn_state.diver_array
        )

        return new_spawn_state, new_shark_positions, new_sub_positions


    def continue_spawn_cycle(i: int, carry):
        spawn_state, shark_positions, sub_positions = carry

        # get the relevant missing entities for this lane from the to_be_spawned array
        missing_entities = jax.lax.dynamic_slice(
            spawn_state.to_be_spawned,
            (i * 3,),
            (3,)
        )

        #print("missing_entities", missing_entities)

        # check in which direction we are moving by finding the first non-zero value in the missing_entities array
        moving_left = jnp.where(
            missing_entities[0] == 0,
            jnp.where(
                missing_entities[1] == 0,
                jnp.where(
                    missing_entities[2] == -1,
                    True,
                    False
                ),
                jnp.where(
                    missing_entities[1] == -1,
                    True,
                    False
                )
            ),
            jnp.where(
                missing_entities[0] == -1,
                True,
                False
            )
        )

        #print("moving_left", moving_left)

        # Find the index of the first non-zero value based on direction
        def scan_right_to_left(j, val):
            return jnp.where(
                missing_entities[j] != 0,
                j,
                val
            )

        def scan_left_to_right(j, val):
            return jnp.where(
                missing_entities[2 - j] != 0,
                2 - j,
                val
            )

        # Use fori_loop to scan array in appropriate direction
        spawn_idx = jax.lax.cond(
            moving_left,
            lambda _: jax.lax.fori_loop(0, 3, scan_left_to_right, -1),
            lambda _: jax.lax.fori_loop(0, 3, scan_right_to_left, -1),
            operand=None
        )

        #print("spawn_idx", spawn_idx)

        spawn_idx = spawn_idx.astype(int)

        # Get reference x position from neighboring entity
        # For moving left, look at entity to the right (spawn_idx + 1)
        # For moving right, look at entity to the left (spawn_idx - 1)
        reference_idx = jnp.where(moving_left, spawn_idx - 1, spawn_idx + 1)
        reference_idx = reference_idx.astype(int)
        base_idx = i * 3  # Base index for this lane's entities

        # Get position from either shark or sub position arrays
        # We'll need to check both since we don't know which type exists
        reference_shark_pos = shark_positions[base_idx + reference_idx]
        reference_sub_pos = sub_positions[base_idx + reference_idx]
        #print("shark_pos", shark_positions)
        #print("reference_idx", reference_idx)
        #print("reference_shark_pos", reference_shark_pos)

        # Use whichever position is non-zero (active)
        reference_x = jnp.where(
            reference_shark_pos[0] != 0,
            reference_shark_pos[0],
            reference_sub_pos[0]
        )

        #print("reference_x", reference_x)

        edge_case = reference_x == 0
        # Edge Case: third option exists for the pattern 1 0 1, then check the next entity
        edge_case_reference_idx = jnp.where(moving_left, spawn_idx - 2, spawn_idx + 2)

        edge_case_reference_idx = edge_case_reference_idx.astype(int)

        reference_x = jnp.where(
            edge_case,
            jnp.where(
                shark_positions[base_idx + edge_case_reference_idx][0] != 0,
                shark_positions[base_idx + edge_case_reference_idx][0],
                sub_positions[base_idx + edge_case_reference_idx][0]
            ),
            reference_x
        )

        # Get base spawn position for this lane
        base_spawn_pos = get_spawn_position(moving_left, jnp.array(i))

        # check if the base spawn position x is 16 / 32 pixels away from the reference x position (depending on the edge case pattern)
        # if yes, spawn the entity, if no, do nothing
        offset = jnp.where(edge_case, 32, 16)
        should_spawn = jnp.abs(base_spawn_pos[0] - reference_x) >= offset

        spawn_pos = jnp.where(
            should_spawn,
            base_spawn_pos,
            jnp.zeros(3)
        )

        # Update positions based on enemy type
        new_shark_positions = shark_positions.at[base_idx + spawn_idx].set(
            jnp.where(jnp.logical_not(spawn_state.prev_sub[i]), spawn_pos, shark_positions[base_idx + spawn_idx])
        )
        new_sub_positions = sub_positions.at[base_idx + spawn_idx].set(
            jnp.where(spawn_state.prev_sub[i], spawn_pos, sub_positions[base_idx + spawn_idx])
        )

        # Update the to_be_spawned array
        #print("should_spawn", should_spawn)
        new_to_be_spawned = spawn_state.to_be_spawned.at[base_idx + spawn_idx].set(
            jnp.where(
                should_spawn,
                jnp.array(0),  # Single value
                missing_entities[spawn_idx]
            )
        )
        #print("new_to_be_spawned", new_to_be_spawned)

        # Then create the new spawn state with the updated array
        new_spawn_state = SpawnState(
            difficulty=spawn_state.difficulty,
            lane_dependent_pattern=spawn_state.lane_dependent_pattern,
            to_be_spawned=new_to_be_spawned,
            survived=spawn_state.survived,
            prev_sub=spawn_state.prev_sub,
            spawn_timers=spawn_state.spawn_timers,
            diver_array=spawn_state.diver_array
        )

        return new_spawn_state, new_shark_positions, new_sub_positions



    def process_lane(i, carry):
        spawn_state, shark_positions, sub_positions = carry
        base_idx = i * 3  # Base index for this lane's slots

        # determine if we need to initialize a new pattern or keep spawning for the current one
        # do this by checking in the relevant part of the to_be_spawned array if there are still 1s
        relevant_to_be_spawned = jax.lax.dynamic_slice(spawn_state.to_be_spawned, (base_idx,), (3,))

        # if there are still 1s in the relevant part of the to_be_spawned array, keep spawning
        keep_spawning = jnp.any(relevant_to_be_spawned)

        # check the lane spawn timer
        lane_timer = spawn_state.spawn_timers[i]

        lane_empty = jnp.all(
            jnp.array([
            jnp.logical_and(
                is_slot_empty(shark_positions[base_idx + j]),
                is_slot_empty(sub_positions[base_idx + j])
            )
            for j in range(3)
            ])
        )

        # if the lane timer is unequal to 0, continue_spawn_cycle may still be called but initialize_new_spawn_cycle should not be called
        allow_new_initialization = jnp.logical_and(lane_timer == 0, lane_empty)
        ##print("IDX", i)
        ##print("keep_spawning", keep_spawning)
        ##print("relevant_to_be_spawned", relevant_to_be_spawned)

        def handle_no_spawning(x):
            return jax.lax.cond(
                allow_new_initialization,
                lambda y: initialize_new_spawn_cycle(i, y),
                lambda y: (y[0], y[1], y[2]),  # Return unchanged state
                x
            )

        new_spawn_state, new_shark_positions, new_sub_positions = jax.lax.cond(
            keep_spawning,
            lambda x: continue_spawn_cycle(i, x),
            handle_no_spawning,
            (spawn_state, shark_positions, sub_positions)
        )

        return new_spawn_state, new_shark_positions, new_sub_positions



    def lane_needs_update(i, spawn_state, shark_positions, sub_positions):
        base_idx = i * 3  # Base index for this lane's slots

        # get how many entities in this lane are inactive
        lane_empty = jnp.all(
            jnp.array([
                jnp.logical_and(
                    is_slot_empty(shark_positions[base_idx + j]),
                    is_slot_empty(sub_positions[base_idx + j])
                )
                for j in range(3)
            ])
        )

        # check if the to_be_spawned array has any 1s in the relevant part
        relevant_to_be_spawned = jax.lax.dynamic_slice(spawn_state.to_be_spawned, (base_idx,), (3,))

        return jnp.logical_or(
            lane_empty,
            jnp.any(relevant_to_be_spawned)
        )

    # count down the spawn timers
    new_spawn_timers = jnp.where(
        spawn_state.spawn_timers > 0,
        spawn_state.spawn_timers - 1,
        spawn_state.spawn_timers
    )

    new_state = spawn_state._replace(spawn_timers=new_spawn_timers)

    # loop over the 4 lanes, check if they need an update and update them if necessary
    for i in range(4):
        ready_lane_idx = jax.lax.cond(
            lane_needs_update(i, new_state, shark_positions, sub_positions),
            lambda x: i,
            lambda x: -1,
            (new_state, shark_positions, sub_positions)
        )

        new_state, shark_positions, sub_positions = jax.lax.cond(
            ready_lane_idx >= 0,
            lambda x: process_lane(i, x),
            lambda x: x,
            (new_state, shark_positions, sub_positions)
        )

    return new_state, shark_positions, sub_positions

def step_enemy_movement(spawn_state: SpawnState,
                       shark_positions: chex.Array,
                       sub_positions: chex.Array,
                       step_counter: chex.Array) -> Tuple[chex.Array, chex.Array, SpawnState]:
    """Update enemy positions based on their patterns"""

    def should_move_x(step_counter, difficulty):
        # Base pattern length (8 or 16 steps)
        cycle_len = jnp.where(
            jnp.isin(difficulty, jnp.array([0, 2, 4, 6, 8, 12])),
            8,
            16
        )

        cycle_step = step_counter % cycle_len

        # For constant movement at difficulty 10
        constant_movement = difficulty == 10

        # For difficulties 11+: movement with acceleration
        is_accelerated = difficulty > 10
        accel_spacing = cycle_len // (jnp.minimum(difficulty - 10, 8) + 1)
        is_accel_point = (cycle_step + 1) % accel_spacing == 0
        accelerated_movement = jnp.where(is_accel_point, 2, 1)

        # For early difficulties (0-9)
        # Now starts much slower:
        base_frequency = 4 - (difficulty // 3)  # Starts at 4 and decreases
        move_frequency = jnp.maximum(base_frequency, 1)  # Never slower than 1
        basic_movement = (cycle_step % move_frequency) == 0

        # Combine all cases using where
        movement = jnp.where(
            constant_movement,
            True,
            jnp.where(
                is_accelerated,
                accelerated_movement,
                basic_movement
            )
        )

        return movement

    def get_shark_offset(step_counter): # shark offset should be constant over the difficulty levels..
        phase = step_counter // 4
        cycle_position = phase % 32

        raw_offset = jnp.where(
            cycle_position < 16,
            cycle_position // 2,  # 0->7
            7 - (cycle_position - 16) // 2  # 7->0
        )

        return raw_offset - 4

    def move_enemy(pos, is_shark, difficulty, slot_idx):
        is_active = jnp.logical_not(is_slot_empty(pos))
        moving_left = pos[2] < 0

        # X movement
        should_move_now = should_move_x(step_counter, difficulty)
        velocity_x = jnp.where(moving_left, -1, 1)
        movement_x = jnp.where(should_move_now, velocity_x, 0)

        # Base Y position comes from spawn positions
        base_y = SPAWN_POSITIONS_Y[slot_idx // 3]  # Divide by 3 to get lane index

        # Calculate Y position
        y_position = jnp.where(
            is_shark,
            base_y + get_shark_offset(step_counter),
            base_y - SUBMARINE_Y_OFFSET
        )

        # Apply movements
        new_pos = jnp.where(
            is_active,
            jnp.array([pos[0] + movement_x, y_position, pos[2]]),
            pos
        )

        # Check bounds and update survived status
        out_of_bounds = jnp.logical_or(new_pos[0] < -8, new_pos[0] >= 168)

        # if it is out of bounds, it is not active anymore. Set to 0,0,0
        new_pos = jnp.where(
            out_of_bounds,
            jnp.zeros_like(new_pos),
            new_pos
        )

        return new_pos, out_of_bounds # out of bounds is the same as survived


    new_shark_positions = jnp.zeros_like(shark_positions)
    new_sub_positions = jnp.zeros_like(sub_positions)
    new_survived = spawn_state.survived

    for i in range(len(shark_positions)):
        new_pos, survived = move_enemy(shark_positions[i], True, spawn_state.difficulty, i)
        new_shark_positions = new_shark_positions.at[i].set(new_pos)
        new_survived = jnp.where(
            survived,
            new_survived.at[i].set(True),
            new_survived
        )

    for i in range(len(sub_positions)):
        new_pos, survived = move_enemy(sub_positions[i], False, spawn_state.difficulty, i)
        new_sub_positions = new_sub_positions.at[i].set(new_pos)
        new_survived = jnp.where(
            survived,
            new_survived.at[i].set(True),
            new_survived
        )

    # Update spawn state with new survived status
    new_spawn_state = spawn_state._replace(survived=new_survived)

    return new_shark_positions, new_sub_positions, new_spawn_state


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

        base_idx = i * 3  # Base index for this lane's slots

        # Get current diver position and enemy positions for this lane
        diver_pos = positions[i]

        # Check if a diver exists in this slot
        diver_exists = diver_pos[2] != 0

        # check if there is any enemy in the lane
        lane_empty = jnp.all(
            jnp.array([
                jnp.logical_and(
                    is_slot_empty(shark_positions[base_idx + j]),
                    is_slot_empty(sub_positions[base_idx + j])
                )
                for j in range(3)
            ])
        )

        # Check if we should spawn based on if a diver exists and the lane is empty and random chance (semi random, use step_counter)
        random_should_spawn = step_counter % 58 == 0 # idk

        # TODO: does not work, idk why
        should_spawn = jnp.logical_and(jnp.logical_not(diver_exists), jnp.logical_and(lane_empty, random_should_spawn))  # Only spawn if no sub and no existing diver

        x_pos = jnp.where(left, 168, 0)
        direction = jnp.where(left, -1, 1)

        # Spawn diver with direction matching enemies
        new_diver = jnp.where(
            should_spawn,
            jnp.array([x_pos, DIVER_SPAWN_POSITIONS[i], direction]),
            diver_pos
        )

        # Return updated carry tuple
        return left, positions.at[i].set(new_diver)

    # Initialize RNG (keeping for pattern consistency)
    rng = jax.random.PRNGKey(42)
    rng, diver_rng = jax.random.split(rng)
    left = jax.random.bernoulli(diver_rng, 0.3)

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

        curr_range = jnp.array([i * 3, i * 3 + 1, i * 3 + 2])
        # get the three sharks in the lane
        all_shark_lane_pos = shark_positions[curr_range]

        # Get shark in the same lane for collision check
        shark_lane_pos = get_front_entity(i, all_shark_lane_pos)
        shark_collision = jnp.logical_and(
            is_active,
            check_collision(
                jnp.array([shark_lane_pos[0], shark_lane_pos[1]]),
                SHARK_SIZE,
                jnp.array([diver_pos[0], diver_pos[1]]),
                DIVER_SIZE
            )
        )

        # TODO: sometimes divers teleport infront of sharks, reason unkown

        # check in which direction the shark is moving and copy the direction to the diver (even if it is not moving)
        direction_of_shark = jnp.where(
            shark_lane_pos[2] == 0,
            diver_pos[2],
            shark_lane_pos[2]
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
            jnp.array([new_x, DIVER_SPAWN_POSITIONS[i], direction_of_shark])
        )

        # Update collection count if collected
        new_collected = collected + jnp.where(should_collect, 1, 0)

        # Update the diver position and collection count
        return positions.at[i].set(new_pos), new_collected

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
    new_shark_positions, new_sub_positions, spawn_state_after_movement = step_enemy_movement(
        spawn_state,
        shark_positions,
        sub_positions,
        state.step_counter
    )

    # Update spawns using updated spawn state
    new_spawn_state, new_shark_positions, new_sub_positions = update_enemy_spawns(
        spawn_state_after_movement,
        new_shark_positions,
        new_sub_positions,
        state.step_counter
    )

    # spawn new divers
    diver_positions = spawn_divers(new_spawn_state, state.diver_positions, new_shark_positions, new_sub_positions, state.step_counter)

    return new_spawn_state, new_shark_positions, new_sub_positions, diver_positions


def surface_sub_step(state: State) -> chex.Array:
    # Check direction value specifically to get scalar boolean
    sub_exists = state.surface_sub_position[2] != 0

    def spawn_sub(_):
        return jnp.array([159, 45, -1])  # Always spawns right facing left

    def move_sub(carry):
        sub_pos = carry
        new_x = jnp.where(
            state.step_counter % 4 == 0,
            sub_pos[0] - 1,  # Direction always -1
            sub_pos[0]
        )

        # Return either zeros or new position
        return jnp.where(
            jnp.logical_or(new_x < -8, sub_pos[2] == 0),
            jnp.zeros(3),
            jnp.array([new_x, 45, -1])
        )

    # Each condition needs to be scalar
    enough_rescues = state.successful_rescues >= 2
    enough_divers = state.divers_collected >= 1
    correct_timing = jnp.logical_and(state.step_counter % 256 == 0, state.step_counter != 0)

    # check if the submarine should spawn
    should_spawn = jnp.logical_and(
        jnp.logical_and(enough_rescues, enough_divers),
        jnp.logical_and(correct_timing, ~sub_exists)
    )

    temp1 = spawn_sub(state.surface_sub_position)
    temp2 = move_sub(state.surface_sub_position)

    return jnp.where(
        should_spawn,
        temp1,
        temp2
    )

def enemy_missiles_step(curr_sub_positions, curr_enemy_missile_positions, step_counter) -> chex.Array:
    def single_missile_step(i, carry):
        # Input i is the loop index, carry is the full array of missile positions
        # Get current submarine and missile for this index
        missile_pos = carry[i]

        # get the current range of the submarines in the lane
        range_start = i * 3
        current_sub_idx = jnp.array([range_start, range_start + 1, range_start + 2])
        lane_subs = curr_sub_positions[current_sub_idx]

        # get the position of the front submarine (thats the only relevant one)
        sub_pos = get_front_entity(i, lane_subs)

        # check if the missile is in frame
        missile_exists = missile_pos[2] != 0

        # check if the missile should be spawned
        should_spawn = jnp.logical_and(
            jnp.logical_not(missile_exists),
            jnp.logical_and(
                sub_pos[0] >= MISSILE_SPAWN_POSITIONS[0],
                sub_pos[0] <= MISSILE_SPAWN_POSITIONS[1]
            )
        )

        # Calculate new missile position ( x -/+ 4 (depending on direction), y = 47, direction = sub direction)
        new_missile_x = jnp.where( # could be sub_pos[0] + 4 * sub_pos[2] as well, but this is easier to read
            sub_pos[2] == 1,
            sub_pos[0] + 4,
            sub_pos[0] - 4
        )

        new_missile = jnp.where(
            should_spawn,
            jnp.array([new_missile_x, ENEMY_MISSILE_Y[i], sub_pos[2]]),  # Use submarine's direction
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
        0, 4,
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

    # if the player shot and there is no missile in frame, then we can shoot a missile
    # the missile y is the current player y position + 7
    # the missile x is either player x + 3 if facing left or player x + 13 if facing right
    new_missile = jnp.where(
        jnp.logical_and(fire, jnp.logical_not(missile_exists)),
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
    has_divers = state.divers_collected >= 0  # Changed to > 0 instead of >= 0
    has_all_divers = state.divers_collected >= 6
    needs_oxygen = state.oxygen < 64

    # Special handling for initialization state
    in_init_state = state.just_surfaced == -1
    started_diving = player_y > PLAYER_START_Y
    filling_init_oxygen = jnp.logical_and(in_init_state, state.oxygen < 64)

    # Surfacing conditions
    increase_ox = jnp.logical_and(at_surface, needs_oxygen)
    stay_same = jnp.logical_and(player_y >= PLAYER_BREATHING_Y[0],
                                player_y <= PLAYER_BREATHING_Y[1])

    # Calculate new divers count before other logic
    new_divers_collected = jnp.where(
        jnp.logical_and(just_surfaced, has_divers),
        jnp.where(
            in_init_state,
            state.divers_collected,
            state.divers_collected - 1
        ),
        state.divers_collected
    )

    # Handle surfacing without divers - prevent during init
    # Only lose life if we started with no divers
    lose_life = jnp.logical_and(
        jnp.logical_and(just_surfaced, new_divers_collected < 0),
        jnp.logical_not(in_init_state)
    )

    # Handle surfacing with all divers
    should_reset = jnp.logical_and(just_surfaced, has_all_divers)

    # Update surfacing flag with consideration for remaining divers
    new_just_surfaced = jnp.where(
        in_init_state,
        jnp.where(
            jnp.logical_and(started_diving, state.oxygen >= 63),
            jnp.array(0),
            jnp.array(-1)
        ),
        jnp.where(
            was_underwater,
            jnp.array(0),
            jnp.where(
                at_surface,
                jnp.array(1),
                state.just_surfaced
            )
        )
    )

    # Handle oxygen changes
    new_oxygen = jnp.where(
        filling_init_oxygen,
        jnp.where(
            state.step_counter % 2 == 0,
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

    # Important: Base blocking decision on has_divers instead of still_has_divers
    can_refill = jnp.logical_and(increase_ox, has_divers)
    new_oxygen = jnp.where(
        jnp.logical_and(can_refill, jnp.logical_not(in_init_state)),
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

    # Increase difficulty when reaching max oxygen after surfacing
    old_difficulty = state.spawn_state.difficulty
    reached_max = jnp.logical_and(jnp.logical_and(new_oxygen >= 64, state.oxygen < 64), jnp.logical_not(in_init_state))
    new_difficulty = jnp.where(reached_max,
                           old_difficulty + 1,
                           old_difficulty)

    new_oxygen = jnp.where(
        stay_same,
        state.oxygen,
        new_oxygen
    )

    # Use has_divers for blocking decision and combine with oxygen check
    should_block = jnp.logical_and(
        at_surface,
        needs_oxygen
    )

    player_x = jnp.where(
        should_block,
        state.player_x,
        player_x
    )

    player_y = jnp.where(
        should_block,
        jnp.array(46, dtype=jnp.int32),  # Force to exact surface position
        player_y
    )

    player_missile_position = jnp.where(
        should_block,
        jnp.zeros(3),
        player_missile_position
    )

    # Prevent oxygen depletion during init
    oxygen_depleted = jnp.logical_and(
        new_oxygen <= jnp.array(0),
        jnp.logical_not(in_init_state)
    )

    return (new_oxygen, player_x, player_y, player_missile_position,
            oxygen_depleted, lose_life, new_divers_collected, should_reset, new_just_surfaced, new_difficulty)

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

def calculate_kill_points(successful_rescues: chex.Array) -> chex.Array:
    """Calculate the points awarded for killing a shark or submarine. Sharks and submarines are worth 20 points.
    The points are increased by 10 for each successful rescue with a maximum of 90."""
    base_points = 20
    max_points = 90
    additional_points = 10 * successful_rescues
    return jnp.minimum(base_points + additional_points, max_points)

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
            surface_sub_position=jnp.zeros(3),  # 1 surface sub
            player_missile_position=jnp.zeros(3),  # x,y,direction
            step_counter=jnp.array(0),
            just_surfaced=jnp.array(-1),
            successful_rescues=jnp.array(0),
            death_counter=jnp.array(0)
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: chex.Array) -> State:
        # First handle death animation if active
        def handle_death_animation():
            # Calculate new positions with frozen X coordinates
            shark_y_positions, _, _ = step_enemy_movement(
                state.spawn_state,
                state.shark_positions,
                state.sub_positions,
                state.step_counter
            )

            # Keep X positions from original state, only update Y
            new_shark_positions = state.shark_positions.at[:, 1].set(shark_y_positions[:, 1])
            should_hide_player = state.death_counter <= 45

            # Return either final reset or animation frame
            return jax.lax.cond(
                state.death_counter <= 1,
                lambda _: self.reset()._replace(
                    lives=state.lives - 1,
                    score=state.score,
                    successful_rescues=state.successful_rescues,
                    divers_collected=jnp.maximum(state.divers_collected - 1, 0),
                    spawn_state=soft_reset_spawn_state(state.spawn_state)
                    # Lose one diver only at end of animation
                ),
                lambda _: state._replace(
                    death_counter=state.death_counter - 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    player_x=jnp.where(should_hide_player, -100, state.player_x),
                    step_counter=state.step_counter + 1
                ),
                operand=None
            )

        def handle_score_freeze():
            # on scoring, the death counter will be set to -(oxygen * 2 + 16 * 6)
            # thats when we get in here, so duplicate the death animation pattern, but decrease the oxygen until its 0
            # Calculate new positions with frozen X coordinates
            shark_y_positions, _, _ = step_enemy_movement(
                state.spawn_state,
                state.shark_positions,
                state.sub_positions,
                state.step_counter
            )

            # Keep X positions from original state, only update Y
            new_shark_positions = state.shark_positions.at[:, 1].set(shark_y_positions[:, 1])

            # calculate the new oxygen
            new_ox = jnp.where(state.death_counter % 2 == 0, state.oxygen - 1, state.oxygen)

            new_ox = jnp.where(
                new_ox <= 0,
                jnp.array(0),
                state.oxygen
            )

            # Return either final reset or animation frame
            return jax.lax.cond(
                state.death_counter >= -1,
                lambda _: self.reset()._replace(
                    score=state.score,
                    successful_rescues=state.successful_rescues + 1,
                    divers_collected=jnp.array(0),
                    spawn_state=soft_reset_spawn_state(state.spawn_state),
                    surface_sub_position=state.surface_sub_position,
                    oxygen=jnp.array(0)
                ),
                lambda _: state._replace(
                    death_counter=state.death_counter + 1,
                    shark_positions=new_shark_positions,
                    sub_positions=state.sub_positions,
                    enemy_missile_positions=state.enemy_missile_positions,
                    player_missile_position=jnp.zeros(3),
                    step_counter=state.step_counter + 1,
                    oxygen=new_ox
                ),
                operand=None
            )

        # Normal game logic starts here
        def normal_game_step():
            # First check if player should be frozen for oxygen refill
            at_surface = state.player_y == 46
            needs_oxygen = state.oxygen < 64
            should_block = jnp.logical_and(at_surface, needs_oxygen)

            # If blocked, force position and disable actions
            player_x = jnp.where(should_block, state.player_x, state.player_x)
            player_y = jnp.where(should_block, jnp.array(46, dtype=jnp.int32), state.player_y)
            action_mod = jnp.where(should_block, jnp.array(NOOP), action)

            # Now calculate movement using potentially modified positions and action
            next_x, next_y, player_direction = player_step(state._replace(player_x=player_x, player_y=player_y),
                                                           action_mod)
            player_missile_position = player_missile_step(state, next_x, next_y, action_mod)

            # Rest of oxygen handling and game logic
            new_oxygen, player_x, player_y, player_missile_position, oxygen_depleted, lose_life_surfacing, new_divers_collected, should_reset, new_just_surfaced, new_difficulty = update_oxygen(
                state, next_x, next_y, player_missile_position
            )

            # Update divers collected count from oxygen mechanics
            state_updated = state._replace(divers_collected=new_divers_collected)

            # update the spawn state with the new difficulty
            new_spawn_state = state_updated.spawn_state._replace(difficulty=new_difficulty)

            # Check missile collisions
            player_missile_position, new_shark_positions, new_sub_positions, new_score, updated_spawn_state = check_missile_collisions(
                player_missile_position,
                state_updated.shark_positions,
                state_updated.sub_positions,
                state_updated.score,
                state_updated.successful_rescues,
                new_spawn_state
            )

            # perform all necessary spawn steps
            new_spawn_state, new_shark_positions, new_sub_positions, new_diver_positions = spawn_step(
                state_updated,
                updated_spawn_state,
                new_shark_positions,
                new_sub_positions,
                state.diver_positions,
            )

            new_diver_positions, new_divers_collected = step_diver_movement(
                new_diver_positions,
                new_shark_positions,
                player_x,
                player_y,
                state_updated.divers_collected,
                state_updated.step_counter,
            )

            new_surface_sub_pos = surface_sub_step(
                state_updated
            )

            state_updated._replace(surface_sub_position=new_surface_sub_pos)

            # update the enemy missile positions
            new_enemy_missile_positions = enemy_missiles_step(
                new_sub_positions,
                state_updated.enemy_missile_positions,
                state_updated.step_counter
            )

            # append the surface submarine to the other submarines for the collision check
            # check if the player has collided with any of the enemies
            player_collision, collision_points = check_player_collision(
                player_x,
                player_y,
                new_sub_positions,
                new_shark_positions,
                new_surface_sub_pos,
                state_updated.enemy_missile_positions,
                new_score,
                state_updated.successful_rescues
            )

            lose_life = jnp.any(jnp.array([oxygen_depleted, player_collision, lose_life_surfacing]))

            # Start death animation but keep divers intact during animation
            death_animation_state = state_updated._replace(
                score=state.score + collision_points,
                death_counter=jnp.array(90),
                spawn_state=soft_reset_spawn_state(state_updated.spawn_state)
            )

            # Calculate points for rescuing divers. Each diver is worth 50 points.
            # Each successful rescue adds 50 points with a maximum of 1000 points each.
            base_points_per_diver = 50
            max_points_per_diver = 1000
            additional_points_per_rescue = 50 * state.successful_rescues
            points_per_diver = jnp.minimum(base_points_per_diver + additional_points_per_rescue, max_points_per_diver)
            total_diver_points = points_per_diver * state.divers_collected

            # Calculate bonus points for remaining oxygen
            oxygen_bonus = state.oxygen * 20

            # Calculate total points for successful rescue
            total_rescue_points = total_diver_points + oxygen_bonus

            # TODO: somewhere the oxygen is depleted on surfacing, this currently blocks the slow draining of oxygen (which is not gameplay relevant -> low priority)
            # scoring freeze, 16 ticks per diver i.e. 6 * 16 and also 2 ticks per remaining oxygen (which is drained!)
            # Create the scoring state
            scoring_state = state_updated._replace(
                lives=state_updated.lives,
                score=state_updated.score + total_rescue_points,
                successful_rescues=state_updated.successful_rescues + 1,
                spawn_state = soft_reset_spawn_state(state_updated.spawn_state),
                death_counter=jnp.array(-(96 + state_updated.oxygen * 2))
            )

            # cap the step counter to 1024
            new_step_counter = jnp.where(
                state_updated.step_counter == 1024,
                jnp.array(0),
                state_updated.step_counter + 1
            )

            # Create the normal returned state
            normal_returned_state = State(
                player_x=player_x,
                player_y=player_y,
                player_direction=player_direction,
                oxygen=new_oxygen,
                divers_collected=new_divers_collected,
                score=new_score,
                lives=state_updated.lives,
                spawn_state=new_spawn_state,
                diver_positions=new_diver_positions,
                shark_positions=new_shark_positions,
                sub_positions=new_sub_positions,
                enemy_missile_positions=new_enemy_missile_positions,
                surface_sub_position=new_surface_sub_pos,
                player_missile_position=player_missile_position,
                step_counter=new_step_counter,
                just_surfaced=new_just_surfaced,
                successful_rescues=state_updated.successful_rescues,
                death_counter=jnp.array(0)
            )


            # First handle surfacing with all divers (scoring)
            intermediate_state = jax.lax.cond(
                should_reset,
                lambda _: scoring_state,
                lambda _: normal_returned_state,
                operand=None
            )

            # Then handle life loss - start death animation instead of immediate reset
            final_state = jax.lax.cond(
                lose_life,
                lambda _: death_animation_state,
                lambda _: intermediate_state,
                operand=None
            )

            # Check for additional life every 10,000 points
            additional_lives = (final_state.score // 10000) - (state.score // 10000)
            new_lives = jnp.minimum(final_state.lives + additional_lives, 6)

            # Update the final state with new lives
            final_state = final_state._replace(lives=new_lives)

            # Check if the game is over
            game_over = final_state.lives <= -1

            # Handle game over state
            return jax.lax.cond(
                game_over,
                lambda _: self.reset()._replace(score=final_state.score, lives=jnp.array(-1)),
                lambda _: final_state,
                operand=None
            )

        # Choose between death animation and normal game step
        return jax.lax.cond(
            state.death_counter > 0,
            lambda _: handle_death_animation(),
            lambda _: jax.lax.cond(
                state.death_counter < 0,
                lambda _: handle_score_freeze(),
                lambda _: normal_game_step(),
                operand=None
            ),
            operand=None
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
        surface_gradient = pygame.Surface((WINDOW_WIDTH, 46))
        for i in range(46):
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

    def draw_score_lives_and_divers(self, score, lives, divers):
        """Draw score, lives counter and collected divers"""
        # Draw score on left
        score_text = self.font.render(str(int(score)), True, SCORE_COLOR)
        self.screen.blit(score_text, (10, 10))

        # Draw lives in middle
        lives_text = self.font.render(f"Lives: {int(lives)}", True, SCORE_COLOR)
        self.screen.blit(lives_text, (WINDOW_WIDTH - 200, 10))

        # Draw diver count on right
        divers_text = self.font.render(f"Divers: {int(divers)}/6", True, SCORE_COLOR)
        self.screen.blit(divers_text, (WINDOW_WIDTH - 100, 40))

    def draw_enemies(self, shark_positions, sub_positions, surface_sub_position, difficulty):
        """Draw sharks and submarines"""
        # Draw sharks
        for pos in shark_positions:
            if pos[0] > 0:  # Only draw if x position is valid
                shark_rect = pygame.Rect(
                    int(pos[0]) * 3, int(pos[1]) * 3,
                    SHARK_SIZE[0] * 3, SHARK_SIZE[1] * 3
                )

                color = SHARK_DIFFICULTY_COLORS[difficulty % len(SHARK_DIFFICULTY_COLORS)]


                pygame.draw.rect(self.screen, color, shark_rect)

        # Draw submarines
        for pos in sub_positions:
            if pos[0] > 0:  # Only draw if x position is valid
                sub_rect = pygame.Rect(
                    int(pos[0]) * 3, int(pos[1]) * 3,
                    ENEMY_SUB_SIZE[0] * 3, ENEMY_SUB_SIZE[1] * 3
                )
                pygame.draw.rect(self.screen, ENEMY_SUB_COLOR, sub_rect)

        # draw surface sub
        if surface_sub_position[0] > 0:
            surface_sub_rect = pygame.Rect(
                int(surface_sub_position[0]) * 3, int(surface_sub_position[1]) * 3,
                ENEMY_SUB_SIZE[0] * 3, ENEMY_SUB_SIZE[1] * 3
            )
            pygame.draw.rect(self.screen, ENEMY_SUB_COLOR, surface_sub_rect)

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
        """Draw missiles
        Args:
            missile_positions: Array of shape (N, 3) or (3,) containing missile data
                where each missile has [x, y, direction]
        """
        # check if there is only a single missile (i.e. not multiple 3 element arrays)
        if len(missile_positions.shape) == 1:
            # Handle single missile case - reshape to match expected dimensions
            missile_positions = jnp.expand_dims(missile_positions, axis=0)

        """Draw missiles"""
        for pos in missile_positions:
            # Only draw if missile exists (x position > 0)
            if pos[0] > 0:
                missile_rect = pygame.Rect(
                    int(pos[0]) * 3,  # x position
                    int(pos[1]) * 3,  # y position
                    MISSILE_SIZE[0] * 3,
                    MISSILE_SIZE[1] * 3
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
        self.draw_enemies(state.shark_positions, state.sub_positions, state.surface_sub_position, state.spawn_state.difficulty)
        self.draw_missiles(state.player_missile_position)
        self.draw_missiles(state.enemy_missile_positions)
        self.draw_player(state.player_x, state.player_y)

        # Draw HUD elements
        self.draw_oxygen_bar(state.oxygen)
        self.draw_score_lives_and_divers(state.score, state.lives, state.divers_collected)

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

    # Get jitted functions
    jitted_step = jax.jit(game.step)
    jitted_reset = jax.jit(game.reset)

    curr_state = jitted_reset()

    # Game loop with rendering
    running = True
    frame_by_frame = False
    frameskip = game.frameskip
    counter = 1

    renderer = Renderer()

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
        renderer.clock.tick(1024)

    pygame.quit()
