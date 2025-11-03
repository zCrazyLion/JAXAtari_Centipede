from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.kangaroo_mod_plugins import NoMonkeyMod, NoFallingCoconutMod, PinChildMod

# --- 3. The Registry ---
KANGAROO_MOD_REGISTRY = {
    "no_monkey": NoMonkeyMod,
    "no_falling_coconut": NoFallingCoconutMod,
    "pin_child": PinChildMod,
}

class KangarooEnvMod(JaxAtariModController):
    """
    Game-specific (Group 1) Mod Controller for Kangaroo.
    It inherits all logic from JaxAtariModController and defines
    the REGISTRY.
    """

    REGISTRY = KANGAROO_MOD_REGISTRY

    def __init__(self,
                 env,
                 mods_config: list = [],
                 allow_conflicts: bool = False
                 ):

        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
