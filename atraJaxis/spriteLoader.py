import numpy as np
from atraJaxis.sprite import Sprite
from atraJaxis.renderMode import RenderMode

class SpriteLoader:
    """
    A class for loading and managing sprites and their frames.

    Attributes
    ----------
    frames : dict
        A dictionary to store frames with their names as keys.
    sprites : dict
        A dictionary to store sprites with their names as keys.

    Methods
    -------
    loadFrame(fileName, **kwargs):
        Loads a frame from a .npy file and stores it in the sprite loader.
    loadSprite(name, frames, render_mode):
        Loads a sprite from a list of frame names and a specified render mode.
    getSprite(name, copy=True):
        Retrieves a sprite by name, optionally returning a deep copy.
    """

    def __init__(self):
        """
        Initializes the SpriteLoader instance.
        """
        self.frames = {}
        self.sprites = {}

    def loadFrame(self, fileName, **kwargs):
        """
        Loads a frame from a file and stores it in the sprite loader.

        Parameters
        ----------
        fileName : str
            The file name of the frame image.
        name : str, optional
            The name of the frame. Defaults to the file name if not provided.

        Raises
        ------
        ValueError
            If the frame format is invalid or the name is already used.
        """
        # Load frame (np array) from a .npy file
        frame = np.load(fileName)

        # Check if the frame's shape is [[[r, g, b, a], ...], ...]
        if frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError("Invalid frame format. The frame must have a shape of (height, width, 4).")

        # Name defaults to the file name, or uses the name parameter
        name = kwargs.get('name', fileName)
        if name in self.frames:
            raise ValueError("The frame name is already used.")

        # Store frame
        self.frames[name] = frame

    def loadSprite(self, name, frames, render_mode):
        """
        Loads a sprite from a list of frame names and a render mode.

        Parameters
        ----------
        name : str
            The name of the sprite.
        frames : list of tuples
            A list of tuples (frame_name: str, duration: int) representing the frames of the sprite.
        render_mode : RenderMode
            The render mode of the sprite (RenderMode.LOOP or RenderMode.ONCE).

        Raises
        ------
        ValueError
            If any frame is not loaded or the render mode is invalid.
        """
        # Check if the frames are loaded
        for frame_name, _ in frames:
            if frame_name not in self.frames:
                raise ValueError(f"The frame {frame_name} is not loaded.")

        # Check render_mode is a RenderMode enum
        if not isinstance(render_mode, RenderMode):
            raise ValueError("Invalid render mode. The render_mode must be a RenderMode enum.")

        # Store sprite
        self.sprites[name] = Sprite(
            [(self.frames[frame_name], duration) for frame_name, duration in frames],
            render_mode
        )

    def getSprite(self, name, copy=True):
        """
        Gets a sprite by name.

        Parameters
        ----------
        name : str
            The name of the sprite.
        copy : bool, optional
            Whether to return a deep copy of the sprite. Defaults to True.

        Returns
        -------
        Sprite
            The sprite object.

        Raises
        ------
        ValueError
            If the sprite is not loaded.
        """
        if name not in self.sprites:
            raise ValueError(f"The sprite {name} is not loaded.")

        sprite = self.sprites[name]
        # Return a deep copy or the original sprite
        return Sprite(
            [(frame.copy(), duration) for frame, duration in sprite.key_frames],
            sprite.render_mode
        ) if copy else sprite
