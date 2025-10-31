import importlib
import inspect

from jaxatari.environment import JaxEnvironment
from jaxatari.renderers import JAXGameRenderer
from jaxatari.modification import JaxAtariModWrapper


# Map of game names to their module paths
GAME_MODULES = {
    "pong": "jaxatari.games.jax_pong",
    "seaquest": "jaxatari.games.jax_seaquest",
    "kangaroo": "jaxatari.games.jax_kangaroo",
    "freeway": "jaxatari.games.jax_freeway",
    "breakout": "jaxatari.games.jax_breakout",
    # Add new games here
}

# Mod modules registry: for each game, provide the Controller class path
MOD_MODULES = {
    "pong": "jaxatari.games.mods.pong_mods.PongEnvMod",
    "kangaroo": "jaxatari.games.mods.kangaroo_mods.KangarooEnvMod",
    #"freeway": "jaxatari.games.mods.freeway_mods.FreewayEnvMod",
    #"breakout": "jaxatari.games.mods.breakout_mods.BreakoutEnvMod",
}

def _load_from_string(path: str):
    """Dynamically import an attribute from a module path string."""
    module_path, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def list_available_games() -> list[str]:
    """Lists all available, registered games."""
    return list(GAME_MODULES.keys())

def make(game_name: str, mode: int = 0, difficulty: int = 0) -> JaxEnvironment:
    """
    Creates and returns a JaxAtari game environment instance.
    This is the main entry point for creating environments.

    Args:
        game_name: Name of the game to load (e.g., "pong").
        mode: Game mode.
        difficulty: Game difficulty.

    Returns:
        An instance of the specified game environment.
    """
    if game_name not in GAME_MODULES:
        raise NotImplementedError(
            f"The game '{game_name}' does not exist. Available games: {list_available_games()}"
        )
    
    try:
        # 1. Dynamically load the module
        module = importlib.import_module(GAME_MODULES[game_name])
        
        # 2. Find the correct environment class within the module
        env_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JaxEnvironment) and obj is not JaxEnvironment:
                env_class = obj
                break # Found it
        
        if env_class is None:
            raise ImportError(f"No JaxEnvironment subclass found in {GAME_MODULES[game_name]}")
        
        # 3. Instantiate the class, passing along the arguments, and return it
        # TODO: none of our environments use mode / difficulty yet, but we might want to add it here and in the single envs (note: probably be replaced by mods)
        return env_class()

    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load game '{game_name}': {e}") from e

def make_renderer(game_name: str) -> JAXGameRenderer:
    """
    Creates and returns a JaxAtari game environment renderer.

    Args:
        game_name: Name of the game to load (e.g., "pong").

    Returns:
        An instance of the specified game environment renderer.
    """
    if game_name not in GAME_MODULES:
        raise NotImplementedError(
            f"The game '{game_name}' does not exist. Available games: {list_available_games()}"
        )
    
    try:
        # 1. Dynamically load the module
        module = importlib.import_module(GAME_MODULES[game_name])
        
        # 2. Find the correct environment class within the module
        renderer_class = None
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, JAXGameRenderer) and obj is not JAXGameRenderer:
                renderer_class = obj
                break # Found it

        if renderer_class is None:
            raise ImportError(f"No AXGameRenderer subclass found in {GAME_MODULES[game_name]}")

        # 3. Instantiate the class, passing along the arguments, and return it
        return renderer_class()
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load renderer for '{game_name}': {e}") from e
    

def modify(env: JaxEnvironment, 
           game_name: str, 
           mods_config: list,
           allow_conflicts: bool = False
           ) -> JaxEnvironment:
    """
    Applies a list of modifications to a JaxAtari environment
    using the full two-stage (Controller + Wrapper) pipeline.
    """
    if not mods_config:
        return env
    
    if game_name not in MOD_MODULES:
        raise NotImplementedError(f"No mod module defined for '{game_name}'.")
    
    try:
        # 1. Load the specific controller class (e.g., PongEnvMod)
        ControllerClass = _load_from_string(MOD_MODULES[game_name])

        # 2. BUILD STAGE 1 (Internal Controller)
        # the controller internally filters for internal mods and overrides the methods with the modded ones.
        modded_env = ControllerClass(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts
        )
        
        # 3. BUILD STAGE 2 (Post-Step Wrapper)
        # the wrapper filters for post-step mods and runs their logic after the step is complete (mostly state attribute updates).
        final_env = JaxAtariModWrapper(
            env=modded_env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts
        )
            
        return final_env
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load mods for '{game_name}': {e}") from e