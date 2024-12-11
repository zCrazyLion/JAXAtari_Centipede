import pygame
from ocatari import OCAtari
from ocatari.ram.pong import Ball, Player, Enemy
import matplotlib.pyplot as plt
from jax_pong import Game as JaxPong, Renderer as JaxRenderer, STATE_TRANSLATOR, PLAYER_X, ENEMY_X, get_human_action, \
    WINDOW_WIDTH, WINDOW_HEIGHT
import random
import numpy as np

VISUALIZE = True # Set this to True to visualize the comparison
CONTROL_ACTIONS = True # Set this to True to control the game with the keyboard

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

if VISUALIZE:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH*2, WINDOW_HEIGHT))
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
    screen.blit(pygame.transform.scale(jax_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (0, 0))  # Left half
    screen.blit(pygame.transform.scale(oc_surface, (WINDOW_WIDTH, WINDOW_HEIGHT)), (WINDOW_WIDTH, 0))  # Right half
    pygame.display.flip()
    clock.tick(60)

# Run the comparison loop
for i in range(1000):
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

    # Step both environments
    oc_atari_env.step(oc_action)
    oc_atari_image = oc_atari_env.getScreenRGB()

    jax_state = jax_env.step(jax_state, action)  # Get the state from the JAX environment
    jax_img = jax_renderer.get_rgb_img(jax_state)


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
    img_loss = np.sum(jax_img!= oc_atari_image)
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

    # Render the game states if visualization is enabled
    if VISUALIZE:
        render(jax_img, oc_atari_image)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()


# Calculate averages
average_img_loss = total_img_loss / 1000
average_ball_loss = total_ball_loss / 1000
average_player_loss = total_player_loss / 1000
average_enemy_loss = total_enemy_loss / 1000

# Print results
print(f"Average Pixel Loss: {average_img_loss}")
print(f"Average Ball Position Error: {average_ball_loss}")
print(f"Average Player Paddle Position Error: {average_player_loss}")
print(f"Average Enemy Paddle Position Error: {average_enemy_loss}")

if VISUALIZE:
    pygame.quit()