from atraJaxis_defunct.renderMode import RenderMode
import numpy as np

class Sprite:
    """
    A class representing a sprite, which is a sequence of frames that can be rendered on the screen.

    Attributes
    ----------
    key_frames : list
        A list of tuples where each tuple contains a frame (np.array) and its duration (int).
    render_mode : RenderMode
        The rendering mode of the sprite, either looping or playing once.
    current_frame_idx : int
        The index of the current frame being displayed.
    finished : bool
        Indicates whether the sprite has finished playing (relevant for RenderMode.ONCE).
    frames : list
        A flattened list of all frames, each repeated according to its duration.
    total_length : int
        The total number of frames in the sprite.
    transform : dict
        A dictionary containing transformations like rotation, horizontal flip, and vertical flip.

    Methods
    -------
    update():
        Updates the current frame index based on the rendering mode.
    render():
        Returns the currently rendered frame after applying transformations.
    get_current_frame_idx():
        Returns the index of the current frame.
    """

    def __init__(self, frames, render_mode):
        """
        Initializes the Sprite instance.

        Parameters
        ----------
        frames : list
            A list of tuples, where each tuple contains a frame (np.array) and its duration (int).
        render_mode : RenderMode
            The rendering mode of the sprite (RenderMode.LOOP or RenderMode.ONCE).
        """
        self.key_frames = frames  # Format: [(pixels: np.array, duration: int), ...]
        self.render_mode = render_mode  # Render mode: RenderMode enum
        self.current_frame_idx = 0
        self.finished = False

        # Compute the complete sequence of frames based on durations
        self.frames = []  # Flattened frames list
        for frame, duration in frames:
            self.frames.extend([frame] * duration)
        self.total_length = len(self.frames)

        self.transform = {
            "rotation_ccw": 0,  # Counter-clockwise rotation in multiples of 90 degrees
            "flip_horizontal": False,  # Flip horizontally
            "flip_vertical": False,  # Flip vertically
        }

    def update(self):
        """
        Updates the current frame index based on the rendering mode.

        Raises
        ------
        ValueError
            If an invalid render mode is specified.
        """
        if self.render_mode == RenderMode.LOOP:
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        elif self.render_mode == RenderMode.ONCE:
            self.current_frame_idx += 1
            if self.current_frame_idx >= len(self.key_frames):
                self.finished = True
                self.current_frame_idx = len(self.key_frames) - 1
        else:
            raise ValueError("Invalid render mode. The render_mode must be a RenderMode enum.")

    def render(self):
        """
        Renders the current frame after applying transformations.

        Returns
        -------
        np.ndarray
            The transformed frame.

        Raises
        ------
        ValueError
            If the sprite has finished playing in RenderMode.ONCE.
        """
        if self.finished:
            raise ValueError("The sprite has finished playing.")

        original = self.frames[self.current_frame_idx]
        rendered = original.transpose(1, 0, 2)  # Transpose x and y axes

        # Apply vertical flip
        if self.transform["flip_vertical"]:
            rendered = rendered[:, ::-1, :]

        # Apply horizontal flip
        if self.transform["flip_horizontal"]:
            rendered = rendered[::-1, :, :]

        # Apply counter-clockwise rotation
        for _ in range(self.transform["rotation_ccw"] % 4):
            rendered = rendered.transpose(1, 0, 2)[:, ::-1, :]

        return rendered

    def get_current_frame_idx(self):
        """
        Returns the index of the current frame.

        Returns
        -------
        int
            The current frame index.
        """
        return self.current_frame_idx
