from jaxatari.games.mods.centipede_mod_plugins import SlowSpellMod, RandomMushroomsMod, RandomPlayerMovementMod
from jaxatari.modification import JaxAtariModController


class CentipedeEnvMod(JaxAtariModController):
    """
    Game-specific Mod Controller for Centipede.
    It simply inherits all logic from JaxAtariModController and defines the CENTIPEDE_MOD_REGISTRY.
    """

    REGISTRY = {
        "slow_spell": SlowSpellMod,
        "random_mushrooms": RandomMushroomsMod,
        # "random_player_movement": RandomPlayerMovementMod,
    }

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