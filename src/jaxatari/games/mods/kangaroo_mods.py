from jaxatari.modification import JaxAtariModController
from jaxatari.games.mods.kangaroo_mod_plugins import (
    NoMonkeyMod, NoFallingCoconutMod, NoThrownCoconutMod, AlwaysHighCoconutMod,
    PinChildMod, RenderDebugInfo, ReplaceChildWithMonkeyMod, ReplaceBellWithFlameMod, ReplaceLadderWithRopeMod,
    LethalFlameMod, SpawnOnSecondFloorMod, FlameTrapMod, CenterLaddersMod, InvertLaddersMod,
    FirstLevelOnlyMod, SecondLevelOnlyMod, ThirdLevelOnlyMod, FourLaddersMod
)
# --- 3. The Registry ---
KANGAROO_MOD_REGISTRY = {
    "no_monkey": NoMonkeyMod,
    "no_falling_coconut": NoFallingCoconutMod,
    "no_thrown_coconut": NoThrownCoconutMod,
    "high_thrown_coconuts": AlwaysHighCoconutMod,
    "no_danger": ["no_monkey", "no_falling_coconut"], # bundle into a modpack
    "pin_child": PinChildMod,
    "render_debug_info": RenderDebugInfo,
    "replace_child_with_monkey": ReplaceChildWithMonkeyMod,
    "replace_bell_with_flame": ReplaceBellWithFlameMod,
    "replace_ladder_with_rope": ReplaceLadderWithRopeMod,
    "_lethal_bell": LethalFlameMod,
    "lethal_flame": ["_lethal_bell", "replace_bell_with_flame"], # bundle into a modpack
    "spawn_on_second_floor": SpawnOnSecondFloorMod,
    "_flame_trap": FlameTrapMod,
    "flame_trap": ["_lethal_bell", "replace_bell_with_flame", "_flame_trap"], # modpack
    "center_ladders": CenterLaddersMod,
    "invert_ladders": InvertLaddersMod,
    "four_ladders": FourLaddersMod,
    "first_level_only": FirstLevelOnlyMod,
    "second_level_only": SecondLevelOnlyMod,
    "third_level_only": ThirdLevelOnlyMod,
    
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
                 allow_conflicts: bool = True
                 ):
        super().__init__(
            env=env,
            mods_config=mods_config,
            allow_conflicts=allow_conflicts,
            registry=self.REGISTRY
        )
