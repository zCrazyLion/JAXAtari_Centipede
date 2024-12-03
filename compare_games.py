from jax_pong import Game as JaxPong
from jax_pong import Renderer as JaxRenderer
from jax_pong import STATE_TRANSLATOR
import random
import numpy as np

jax_env = JaxPong(frameskip=1)
jax_renderer = JaxRenderer(STATE_TRANSLATOR)
oc_atari_env = OCAtari("Pong-v4, frameskip=1)

jax_env.reset()
oc_atari_env.reset()

total_loss = 0
for i in range(1000):
   action = random.randint(0, 5)

   oc_atari_env.step(action)
   oc_atari_image = oc_atari_env.getScreenRGB()

   state = jax_env.step(action, state)
   jax_img = JaxRenderer.get_rgb_img(state)

   imloss = np.sum(np.count(jax_img != oc_atari_image))
   total_loss += imloss

print(total_loss/1000)