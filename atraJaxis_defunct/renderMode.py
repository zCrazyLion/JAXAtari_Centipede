from enum import Enum


class RenderMode(Enum):
    """
    An enumeration to define rendering modes for sprites.

    Attributes
    ----------
    LOOP : int
        Indicates the sprite should loop through the frames repeatedly.
    ONCE : int
        Indicates the sprite should play the frames once and then destroy itself.
    """

    LOOP = 1
    ONCE = 2
