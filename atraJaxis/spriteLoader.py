# A utility class to load and manage sprites.
import numpy as np
# import sprite class from sprite.py
from sprite import sprite, renderMode
class spriteLoader:
    def __init__(self):
        self.frames = {}
        self.sprites = {}
    def loadFrame(self, fileName, **kwargs):
        """
        Loads a frame from a file and stores it in the sprite loader.

        Parameters:
            fileName (str): The file name of the frame image.
            name (str): Optional: The name of the frame.
        Returns:
            None
        """
        # load frame (np array) from a .npy file
        frame = np.load(fileName)
        # check if the frame's shape is according to format [[[r, g, b, a], ...], ...]
        if frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError("Invalid frame format. The frame must have a shape of (height, width, 4).")
        
        # name is by default the file name, or use the name parameter
        name = kwargs.get('name', fileName)
        if name in self.frames:
            raise ValueError("The frame name is already used.")
        # store frame
        self.frames[name] = frame
        
    def loadSprite(self, name, frames, render_mode):
        """
        Loads a sprite from a list of frame names and a render mode.
        
        Parameters:
            name (str): The name of the sprite.
            frames (list): A list of tuples (frame_name: str, duration: int) representing the frames of the sprite.
            render_mode (renderMode): The render mode of the sprite.
        
        Returns:
            None
        """
        # check if the frames are loaded
        for frame_name, _ in frames:
            if frame_name not in self.frames:
                raise ValueError("The frame {} is not loaded.".format(frame_name))
        # check render_mode is a renderMode enum
        if not isinstance(render_mode, renderMode):
            raise ValueError("Invalid render mode. The render_mode must be a renderMode enum.")
        # store sprite
        self.sprites[name] = sprite([(self.frames[frame_name], duration) for frame_name, duration in frames], render_mode)
        
    def getSprite(self, name):
        """
        Gets a sprite by name.
        
        Parameters:
            name (str): The name of the sprite.
        
        Returns:
            sprite: The sprite object.
        """
        if name not in self.sprites:
            raise ValueError("The sprite {} is not loaded.".format(name))
        return self.sprites[name]