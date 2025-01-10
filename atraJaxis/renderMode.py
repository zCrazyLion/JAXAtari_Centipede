from enum import Enum

# enum of render mode.
# loop: loop through the frames.
# once: play the frames once, then destroy the sprite.
class RenderMode(Enum):
    LOOP = 1
    ONCE = 2 