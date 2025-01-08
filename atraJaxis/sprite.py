from enum import Enum

# enum of render mode.
# loop: loop through the frames.
# once: play the frames once, then destroy the sprite.
class renderMode(Enum):
    LOOP = 1
    ONCE = 2 

# class of sprites.
# sprites are objects that can be drawn on the screen.
# sprites are defined as a sequence of frames, where each frame is a np array of pixels.
class sprite:
    def __init__(self, frames, render_mode):
        self.frames = frames # frames format: [(pixels: np.array, duration: int), ...]
        self.render_mode = render_mode # render_mode: renderMode enum
        self.current_frame = 0
        self.finished = False
        # compute frame sequence
        self.frame_sequence = [] # frame_sequence format: frame_sequence[i] is the i-th frame
        for i in range(len(frames)):
            for _ in range(frames[i][1]):
                self.frame_sequence.append(frames[i][0])
    def update(self):
        if self.render_mode == renderMode.LOOP:
            self.current_frame = (self.current_frame + 1) % len(self.frames)
        elif self.render_mode == renderMode.ONCE:
            self.current_frame += 1
            if self.current_frame >= len(self.frames):
                self.finished = True
                self.current_frame = len(self.frames) - 1
        else:
            # exception: invalid render mode
            raise ValueError("Invalid render mode. The render_mode must be a renderMode enum.")
    
    def get_current_frame(self):
        if(self.finished):
            raise ValueError("The sprite has finished playing.")
        return self.frame_sequence[self.current_frame]