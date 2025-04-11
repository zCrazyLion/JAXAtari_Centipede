import pygame
from ocatari import OCAtari
from ocatari.ram.pong import Ball, Player, Enemy
import matplotlib.pyplot as plt

from jax_pong import (
    Game as JaxPong,
    Renderer as JaxRenderer,
    STATE_TRANSLATOR,
    PLAYER_X,
    ENEMY_X,
    get_human_action,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)
import random
import numpy as np


TIMESTEPS = 1000
VISUALIZE = True  # Set this to True to visualize the comparison
CONTROL_ACTIONS = True  # Set this to True to control the game with the keyboard
RESTART_WHEN_SCORED = (
    False  # Set this to True to restart the game when a point is scored
)
PLOT = False

# Initialize environments
jax_env = JaxPong(frameskip=1)
jax_renderer = JaxRenderer(STATE_TRANSLATOR)
oc_atari_env = OCAtari("Pong-v4", frameskip=1)

# Reset environments
jax_state = jax_env.reset()
oc_atari_env.reset()

# Initialize variables for comparison
total_img_loss = 0
total_ball_loss = 0
total_player_loss = 0
total_enemy_loss = 0

# PLayer paddle
player_errors = []
jax_player_positions = []
oc_player_positions = []
jax_player_speeds = []
oc_player_speeds = []
actions = []

# Ball
ball_x_errors = []
ball_y_errors = []
ball_xy_errors = []
jax_ball_positions = []
oc_ball_positions = []

previous_oc_player_y = None
previous_jax_player_y = None

if VISUALIZE:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH * 2, WINDOW_HEIGHT))
    pygame.display.set_caption("JAX Pong (left) vs OC Atari Pong (right)")
    clock = pygame.time.Clock()


# Function to extract object positions from the OC Atari environment
def extract_positions_oc(env):
    objects = env.objects
    ball = next((obj.xy for obj in objects if isinstance(obj, Ball)), (0, 0))
    player = next((obj.xy for obj in objects if isinstance(obj, Player)), (0, 0))
    enemy = next((obj.xy for obj in objects if isinstance(obj, Enemy)), (0, 0))
    return ball, player, enemy


# Function to render the game states
def render(jax_img, oc_atari_image):
    jax_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(jax_img, k=1)))
    oc_surface = pygame.surfarray.make_surface(np.flipud(np.rot90(oc_atari_image, k=1)))
    screen.blit(
        pygame.transform.scale(jax_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0)
    )  # Left half
    screen.blit(
        pygame.transform.scale(oc_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)),
        (WINDOW_WIDTH, 0),
    )  # Right half
    pygame.display.flip()
    clock.tick(60)


previous_jax_score = (0, 0)
previous_oc_score = (0, 0)

# Run the comparison loop
for i in range(TIMESTEPS):
    # get an action
    if CONTROL_ACTIONS:
        action = get_human_action()
    else:
        action = random.randint(0, 5)

    # Somehow the actions have to be flipped for the OC Atari environment, #TODO investigate why
    if action == 2:  # Move up
        oc_action = 3  # Move down
    elif action == 3:  # Move down
        oc_action = 2  # Move up
    else:
        oc_action = action

    # Save the action
    actions.append(action)

    # Step both environments
    oc_atari_env.step(oc_action)
    oc_atari_image = oc_atari_env.getScreenRGB()

    jax_state = jax_env.step(
        jax_state, action
    )  # Get the state from the JAX environment
    jax_img = jax_renderer.get_rgb_img(jax_state)

    # Check for score changes
    ram_state = oc_atari_env.get_ram()
    player_score = ram_state[14]
    enemy_score = ram_state[13]
    current_oc_score = (player_score, enemy_score)
    current_jax_score = (jax_state.player_score, jax_state.enemy_score)

    if RESTART_WHEN_SCORED and (
        current_jax_score != previous_jax_score or current_oc_score != previous_oc_score
    ):
        jax_state = jax_env.reset()
        oc_atari_env.reset()
        previous_jax_score = (0, 0)
        previous_oc_score = (0, 0)
    else:
        previous_jax_score = current_jax_score
        previous_oc_score = current_oc_score

    if PLOT:
        if 100 < i < 110:
            print(action)
            a = plt.figure()
            a.add_subplot(1, 2, 1)
            plt.imshow(jax_img)
            plt.title("Jax")
            a.add_subplot(1, 2, 2)
            plt.imshow(oc_atari_image)
            plt.title("OC Atari")
            plt.show()

    # Pixel difference
    img_loss = np.sum(jax_img != oc_atari_image)
    total_img_loss += img_loss

    # Object position difference
    oc_ball, oc_player, oc_enemy = extract_positions_oc(oc_atari_env)

    # Extract positions from JAX state
    jax_ball = (jax_state.ball_x, jax_state.ball_y)
    jax_player = (PLAYER_X, jax_state.player_y)
    jax_enemy = (ENEMY_X, jax_state.enemy_y)

    # Compute positional losses for each object
    ball_loss = np.linalg.norm(np.array(oc_ball) - np.array(jax_ball))
    player_loss = np.linalg.norm(np.array(oc_player) - np.array(jax_player))
    enemy_loss = np.linalg.norm(np.array(oc_enemy) - np.array(jax_enemy))

    # Accumulate losses
    total_ball_loss += ball_loss
    total_player_loss += player_loss
    total_enemy_loss += enemy_loss

    # Save player paddle error and positions
    player_errors.append(player_loss)
    jax_player_positions.append(jax_player[1])
    oc_player_positions.append(oc_player[1])

    if previous_oc_player_y is not None:
        oc_player_speed = oc_player[1] - previous_oc_player_y
        jax_player_speed = jax_player[1] - previous_jax_player_y
    else:
        oc_player_speed = 0
        jax_player_speed = 0
        print("First timestep")
    oc_player_speeds.append(oc_player_speed)
    previous_oc_player_y = oc_player[1]
    jax_player_speeds.append(jax_player_speed)
    previous_jax_player_y = jax_player[1]

    # Save ball positions and errors after timestep 57 (before that the ball is at (-49, 46) in OC Atari) TODO why?
    if i > 57:
        jax_ball_positions.append(jax_ball)
        oc_ball_positions.append(oc_ball)
        ball_x_errors.append(abs(jax_ball[0] - oc_ball[0]))
        ball_y_errors.append(abs(jax_ball[1] - oc_ball[1]))
        ball_xy_errors.append(np.linalg.norm(np.array(jax_ball) - np.array(oc_ball)))

    # Render the game states if visualization is enabled
    if VISUALIZE:
        render(jax_img, oc_atari_image)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


# Calculate averages
average_img_loss = total_img_loss / TIMESTEPS
average_ball_loss = total_ball_loss / TIMESTEPS
average_player_loss = total_player_loss / TIMESTEPS
average_enemy_loss = total_enemy_loss / TIMESTEPS

# Print results
print(f"Average Pixel Loss: {average_img_loss}")
print(f"Average Ball Position Error: {average_ball_loss}")
print(f"Average Player Paddle Position Error: {average_player_loss}")
print(f"Average Enemy Paddle Position Error: {average_enemy_loss}")

# Plot player paddle error
plt.figure()
for i in range(TIMESTEPS):
    if actions[i] == 2:  # Move up
        plt.axvspan(i, i + 1, color="green", alpha=0.1)
    elif actions[i] == 3:  # Move down
        plt.axvspan(i, i + 1, color="red", alpha=0.1)
plt.plot(player_errors)
plt.title("Player Paddle Error Over Time")
plt.xlabel("Timestep")
plt.ylabel("Error")
plt.show()

# Plot player paddle positions
plt.figure()
for i in range(TIMESTEPS):
    if actions[i] == 2:  # Move up
        plt.axvspan(i, i + 1, color="green", alpha=0.1)
    elif actions[i] == 3:  # Move down
        plt.axvspan(i, i + 1, color="red", alpha=0.1)
plt.plot(jax_player_positions, label="JAX Pong Player Position")
plt.plot(oc_player_positions, label="OC Atari Player Position")
plt.title("Player Paddle Position Over Time")
plt.xlabel("Timestep")
plt.ylabel("Position")
plt.legend()
plt.show()


def smoothing(data):
    result = []
    for i in range(1, len(data) - 1):
        if data[i] == 0 and data[i - 1] != 0 and data[i + 1] != 0:
            data[i] = (data[i - 1] + data[i + 1]) / 2
        result.append(data[i])
    return result


# smoothed_oc_player_speeds = smoothing(oc_player_speeds)
# smoothed_jax_player_speeds = smoothing(jax_player_speeds)

# Plot player paddle speeds
plt.figure()
for i in range(TIMESTEPS):
    if actions[i] == 2:  # Move up
        plt.axvspan(i, i + 1, color="green", alpha=0.1)
    elif actions[i] == 3:  # Move down
        plt.axvspan(i, i + 1, color="red", alpha=0.1)
plt.plot(jax_player_speeds, label="JAX Pong Player Speed (Smoothed)")
plt.plot(oc_player_speeds, label="OC Atari Player Speed (Smoothed)")
plt.title("Player Paddle Speed Over Time")
plt.xlabel("Timestep")
plt.ylabel("Speed")
plt.legend()
plt.show()

# Plot ball euclidean errors
plt.figure()
plt.plot(ball_xy_errors, label="Ball XY Position Error")
plt.title("Ball XY Position Error Over Time")
plt.xlabel("Timestep")
plt.ylabel("Error")
plt.legend()
plt.show()

# Plot ball trajectories with color gradients
plt.figure(figsize=(WINDOW_WIDTH / 100, WINDOW_HEIGHT / 100))
jax_ball_positions_np = np.array(jax_ball_positions)
oc_ball_positions_np = np.array(oc_ball_positions)

jax_colors = np.linspace(0, 1, len(jax_ball_positions_np))
plt.scatter(
    jax_ball_positions_np[:, 0],
    jax_ball_positions_np[:, 1],
    c=jax_colors,
    cmap="autumn",
    label="JAX Pong Ball Trajectory",
    alpha=0.3,
)
oc_colors = np.linspace(0, 1, len(oc_ball_positions_np))
plt.scatter(
    oc_ball_positions_np[:, 0],
    oc_ball_positions_np[:, 1],
    c=oc_colors,
    cmap="winter",
    label="OC Atari Ball Trajectory",
    alpha=0.3,
)

plt.gca().invert_yaxis()
plt.title("Ball Trajectory Over Time")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.show()


if VISUALIZE:
    pygame.quit()
