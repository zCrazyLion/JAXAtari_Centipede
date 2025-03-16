import jax

from environment import JaxEnvironment
from jax_pong import Game as JaxPong
from jax_breakout import Game as JaxBreakout
from jax_freeway import FreewayGameLogic as JaxFreeway
from jax_seaquest import JaxSeaquest
from jax_skiing import SkiingGameLogic as JaxSkiing
from jax_tennis import Game as JaxTennis

class JAXAtari:
    def __init__(self, game_name):
        match game_name:
            case "breakout":
                env = JaxBreakout()
            case "freeway":
                env = JaxFreeway()
            case "pong":
                env = JaxPong(frameskip=1)
            case "seaquest":
                env = JaxSeaquest()
            case "skiing":
                env = JaxSkiing()
            case "tennis":
                env = JaxTennis()
            case _:
                raise NotImplementedError(f"The game {game_name} does not exist")
        self.env: JaxEnvironment = env


    def get_init_state(self):
        fn = jax.jit(self.env.reset)
        state, obs = fn()
        return state

    def step_state_only(self, state, action):
        state, obs, reward, done, info = self.env.step(state, action)
        return state

    def step(self, state, action):
        return self.env.step(state, action)
