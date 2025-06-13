import json
import importlib
import os
import inspect
from typing import Optional, Dict, Any, Tuple, Type

import jax
from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import AtraJaxisRenderer

class JAXAtari:
    """Central entry point for JAXAtari environments.
    
    This class provides a unified interface to create JaxAtari environments.
    These base environments should be used with the wrappers in the `jaxatari.wrappers` module.
    """
    
    # Map of game names to their module paths
    GAME_MODULES = {
        "pong": "jaxatari.games.jax_pong",
        "seaquest": "jaxatari.games.jax_seaquest",
        "kangaroo": "jaxatari.games.jax_kangaroo",
        "freeway": "jaxatari.games.jax_freeway",
        # Add new games here
    }
    
    def __init__(self, game_name: str, mode: int = 0, difficulty: int = 0):
        """Initialize a JaxAtari environment.
        
        Args:
            game_name: Name of the game to load (e.g., "pong", "seaquest")
            mode: Game mode (default = 0)
            difficulty: Game difficulty (default = 0)
        """
        self.mode = mode
        self.difficulty = difficulty
        self._init_game(game_name)
        
        # Pre-compile the core functions
        self._jitted_reset = jax.jit(self.env.reset)
        self._jitted_step = jax.jit(self.env.step)
        if self.renderer is not None:
            self._jitted_render = jax.jit(self.env.render)
    
    @classmethod
    def list_available_games(cls) -> list[str]:
        """List all available games."""
        return list(cls.GAME_MODULES.keys())
        
    def _load_game_module(self, game_name: str) -> Tuple[Type[JaxEnvironment], Any]:
        """Dynamically load a game module.
        
        Args:
            game_name: Name of the game to load
            
        Returns:
            Tuple of (environment class, renderer class)
        """
        if game_name not in self.GAME_MODULES:
            raise NotImplementedError(
                f"The game {game_name} does not exist. Available games: {self.list_available_games()}"
            )
            
        try:
            module = importlib.import_module(self.GAME_MODULES[game_name])
            
            # Find the class that inherits from JaxEnvironment
            env_class = None
            renderer_class = None
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj):
                    if issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
                        env_class = obj
                    if issubclass(obj, AtraJaxisRenderer) and obj is not AtraJaxisRenderer:
                        renderer_class = obj
            
            if env_class is None:
                raise ImportError(f"No class found in {self.GAME_MODULES[game_name]} that inherits from JaxEnvironment")
                
            return env_class, renderer_class
            
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load game {game_name}: {str(e)}")
        
    def _init_game(self, game_name: str):
        """Initialize the game environment and renderer."""
        env_class, renderer_class = self._load_game_module(game_name)
        self.env: JaxEnvironment = env_class()  # Parameters not passed yet
        self.renderer = renderer_class() if renderer_class is not None else None

    def reset(self, key=None):
        """Reset the environment."""
        return self._jitted_reset(key)

    def get_init_state(self):
        """Get the initial state of the environment."""
        obs, state = self._jitted_reset()
        return state

    def step(self, state, action):
        """Step the environment."""
        return self._jitted_step(state, action)

    def render(self, state):
        """Render the current state."""
        if self.renderer is None:
            raise NotImplementedError("No renderer available for this environment")
        return self._jitted_render(state)

    def save_state_as_json(self, state, path):
        """Save the current state to a JSON file."""
        state_dict = state._asdict()
        for item in state_dict:
            state_dict[item] = state_dict[item].tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=4)

    def load_state_from_json(self, curr_state, path):
        """Load a state from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        new_state = curr_state.__class__(**state)
        return new_state
    
    def get_action_space(self):
        """Get the action space of the environment."""
        return self.env.get_action_space()
    
    def get_observation_space(self):
        """Get the observation space of the environment."""
        return self.env.get_observation_space()
    
    def obs_to_flat_array(self, obs):
        """Convert the observation to a flat array."""
        return self.env.obs_to_flat_array(obs)