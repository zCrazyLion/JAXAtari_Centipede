import inspect
import jax
import chex
from functools import partial
from collections import defaultdict
from abc import ABC, abstractmethod
from jaxatari.wrappers import JaxatariWrapper

# --- 1. Plugin Base Classes ---
class JaxAtariInternalModPlugin(ABC):
    """
    Base class for an *internal* mod plugin.
    - Patches methods defined on the plugin.
    - Overrides constants and attributes.
    """
    # manually defined list of mods this mod may conflict with. Does not include functional conflicts (i.e. two mods modifying the same method) which are detected automatically.
    conflicts_with: list = []
    # For core.make to override constants at construction time
    constants_overrides: dict = {}
    # For overriding member attributes set in __init__
    attribute_overrides: dict = {}

class JaxAtariPostStepModPlugin(ABC):
    """
    Base class for a *post-step* mod plugin (runs after step).
    """
    # manually defined list of mods this mod may conflict with. Does not include functional conflicts (i.e. two mods modifying the same method) which are detected automatically.
    conflicts_with: list = []
    # For core.make to override constants at construction time
    constants_overrides: dict = {}

    @abstractmethod
    @partial(jax.jit, static_argnums=(0,))
    def run(self, new_state):
        """
        This function is called by the wrapper *after*
        the main step is complete.
        Access the environment via self._env (set by JaxAtariModWrapper).
        """
        raise NotImplementedError

# --- 2. Shared Helper Function ---

def _check_gameplay_conflicts(
    active_plugins: dict, 
    allow_conflicts: bool
):
    """
    Checks for gameplay conflicts (using the 'conflicts_with' list)
    among a list of instantiated plugins.
    """
    active_keys = set(active_plugins.keys())
    all_conflicts = set()
    for plugin in active_plugins.values():
        all_conflicts.update(plugin.conflicts_with)
    
    conflicts_found = active_keys.intersection(all_conflicts)
    if conflicts_found:
        report = f"Two or more of the requested mods were marked as impacting the same gameplay element and conflict with each other: {conflicts_found}"
        if not allow_conflicts:
            raise ValueError(
                f"{report}\n(Pass allow_conflicts=True to ignore this warning.)"
            )
        else:
            print(f"WARNING: {report}. Conflicts ignored.")

# --- 3. Updated ModController ---

class JaxAtariModController:
    """
    A generic Mod Controller that contains all the logic for:
    1. 'Internal' type plugin discovery and validation.
    2. Automated conflict detection and reporting.
    3. Applying 'internal' type patches (method replacement).
    This class is intended to be inherited by game-specific
    controllers (e.g., PongEnvMod, KangarooEnvMod).
    """
    def __init__(self, 
                 env, 
                 mods_config: list,  # Full list of mod keys
                 registry: dict,     # Full game registry
                 allow_conflicts: bool = False
                 ):
        
        self._env = env
        self.registry = registry
        
        active_plugins = {} # {mod_key: instance}
        patch_map = defaultdict(list)
        
        # 1. --- BUILD & FILTER PHASE ---
        internal_mod_keys = []
        for mod_key in mods_config:
            if mod_key not in self.registry:
                raise ValueError(f"Mod '{mod_key}' not recognized.")
            
            plugin_class = self.registry[mod_key]
            
            # This class only cares about Internal plugins
            if issubclass(plugin_class, JaxAtariInternalModPlugin):
                plugin_instance = plugin_class()
                active_plugins[mod_key] = plugin_instance
                internal_mod_keys.append(mod_key)
                
                # Build patch map for functional conflicts
                for fn_name, _ in inspect.getmembers(plugin_instance, predicate=inspect.ismethod):
                    if not fn_name.startswith("__"):
                        patch_map[fn_name].append(mod_key)

                # --- Strict Check (Attributes) ---
                if hasattr(plugin_instance, "attribute_overrides"):
                    for attr_name in plugin_instance.attribute_overrides:
                        if not hasattr(self._env, attr_name):
                            # Strict: Fail on typo or missing attribute
                            raise AttributeError(
                                f"Mod '{mod_key}' tries to override attribute '{attr_name}', "
                                f"but this attribute does not exist on the base environment."
                            )
        if not active_plugins:
            return # No internal mods to apply
        # 2. --- REPORT PHASE (Gameplay Conflicts) ---
        # Use the shared helper
        _check_gameplay_conflicts(
            active_plugins, 
            allow_conflicts
        )
        # 3. --- REPORT PHASE (Functional Conflicts) ---
        conflicts_found = False
        report_lines = []
        for fn_name, mods in patch_map.items():
            if len(mods) > 1:
                conflicts_found = True
                winner = mods[-1]
                report_lines.append(
                    f"  - The function or attribute '{fn_name}' is modified by multiple mods: {mods}. "
                    f"With allow_conflicts=True, the last mod in the list ('{winner}') would take priority."
                )
        
        if conflicts_found:
            full_report = "\n".join(report_lines)
            if not allow_conflicts:
                raise ValueError(
                    f"Functional conflicts detected:\n{full_report}\n"
                    "(Pass allow_conflicts=True to ignore this warning.)"
                )
            else:
                print(f"WARNING: Functional conflicts detected:\n{full_report}")
        # 4. --- APPLY PHASE ---
        for mod_key in internal_mod_keys: # Iterate in order
            plugin = active_plugins[mod_key]

            # Ensure internal plugin methods can access the environment
            plugin._env = self._env

            # --- Apply Attribute Overrides ---
            if hasattr(plugin, "attribute_overrides"):
                for attr_name, value in plugin.attribute_overrides.items():
                    setattr(self._env, attr_name, value)

            # --- Apply Function Patches ---
            for fn_name, fn_logic in inspect.getmembers(plugin, predicate=inspect.ismethod):
                if not fn_name.startswith("__"):
                    # Bind method normally - plugin methods can access self._env as an attribute
                    setattr(self._env, fn_name, fn_logic)
            
    def __getattr__(self, name):
        """Delegates all other attribute and method access to the wrapped environment."""
        return getattr(self._env, name)



# --- 4. ModWrapper for post-step plugins ---
class JaxAtariModWrapper(JaxatariWrapper):
    """
    A generic Mod Wrapper that handles all *post-step* logic.
    """
    def __init__(self, 
                 env, 
                 mods_config: list,  # Full list of mod keys
                 allow_conflicts: bool = False
                 ):
        
        super().__init__(env)
        
        try:
            registry = self._env.registry
        except AttributeError:
            raise TypeError(
                "JaxAtariModWrapper must be applied to a JaxAtariModController, "
                "but the environment provided does not have a .registry attribute."
            )

        # 1. --- BUILD & FILTER PHASE ---
        active_plugins = {} # {mod_key: instance}
        for mod_key in mods_config:
            if mod_key not in registry:
                raise ValueError(f"Mod '{mod_key}' not recognized.")
            
            plugin_class = registry[mod_key]
            
            if issubclass(plugin_class, JaxAtariPostStepModPlugin):
                active_plugins[mod_key] = plugin_class()
        
        if not active_plugins:
            self.post_step_mods = []
            return
        
        # 2. --- REPORT PHASE (Gameplay Conflicts) ---
        # Use the shared helper
        _check_gameplay_conflicts(active_plugins, allow_conflicts)
        
        # 3. --- APPLY PHASE ---
        # Store env reference on each plugin instance
        for plugin in active_plugins.values():
            plugin._env = self._env
        
        # Store the bound run methods (they're already bound to plugin instances)
        self.post_step_mods = [
            plugin.run 
            for plugin in active_plugins.values()
        ]

    # this is explictly defined to show that we do not apply any post-step logic to the reset! This should be fine. Hopefully.
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        return self._env.reset(key)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # 1. Run the (modded) step from the underlying env
        obs, new_state, reward, done, info = self._env.step(state, action)
        
        # 2. Run all post-step mods in order
        for mod_fn in self.post_step_mods:
            new_state = mod_fn(new_state)
            
        return obs, new_state, reward, done, info