from functools import partial

import jax

from jax_pong import Game as JaxPong
from jax_breakout import Game as JaxBreakout
from jax_freeway import FreewayGameLogic as JaxFreeway
from jax_freeway import GameConfig as JaxFreewayConfig
from jax_seaquest import Game as JaxSeaquest
from jax_skiing import SkiingGameLogic as JaxSkiing
from jax_skiing import GameConfig as JaxSkiingConfig
from jax_tennis import Game as JaxTennis

class JAXAtari:
    def __init__(self, game_name):
        match game_name:
            case "breakout":
                env = JaxBreakout()
            case "freeway":
                conf = JaxFreewayConfig()
                env = JaxFreeway(conf)
            case "pong":
                env = JaxPong(frameskip=1)
            case "seaquest":
                env = JaxSeaquest()
            case "skiing":
                conf = JaxSkiingConfig()
                env = JaxSkiing(conf)
            case "tennis":
                env = JaxTennis()
            case _:
                raise NotImplementedError(f"The game {game_name} does not exist")
        self.env = env

    def get_init_state(self):
        fn = jax.jit(self.env.reset)
        return fn()

    def step(self, state, action):
        return self.env.step(state, action)
