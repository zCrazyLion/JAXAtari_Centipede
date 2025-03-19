from ocatari.core import OCAtari
import random

env = OCAtari("ALE/Pong-v5", mode="ram", hud=True, render_mode="rgb_array")
observation, info = env.reset()
action = random.randint(0, env.nb_actions-1)
while(True):
    obs, reward, terminated, truncated, info = env.step(action)