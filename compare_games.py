from ocatari import OCAtari
from ocatari.ram.pong import Ball, Player, Enemy
import matplotlib.pyplot as plt
from jax_pong import Game as JaxPong, Renderer as JaxRenderer, STATE_TRANSLATOR, PLAYER_X, ENEMY_X
import random
import numpy as np

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

# Define a function to extract object positions from the OC Atari environment
def extract_positions_oc(env):
    objects = env.objects
    ball = next((obj.xy for obj in objects if isinstance(obj, Ball)), (0, 0))
    player = next((obj.xy for obj in objects if isinstance(obj, Player)), (0, 0))
    enemy = next((obj.xy for obj in objects if isinstance(obj, Enemy)), (0, 0))
    return ball, player, enemy


# Run the comparison loop
for i in range(1000):
    # Get a random action
    action = random.randint(0, 5)

    # Step both environments
    oc_atari_env.step(action)
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