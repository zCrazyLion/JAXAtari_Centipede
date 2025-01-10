from renderMode import RenderMode

# class of sprites.
# sprites are objects that can be drawn on the screen.
# sprites are defined as a sequence of frames, where each frame is a np array of pixels.
class Sprite:
    def __init__(self, frames, render_mode):
        self.key_frames = frames # key_frames format: [(pixels: np.array, duration: int), ...]
        self.render_mode = render_mode # render_mode: renderMode enum
        self.current_frame_idx = 0
        self.finished = False
        # compute frame sequence
        self.frames = [] # frames format: frames[i] is the i-th frame
        for i in range(len(frames)):
            for _ in range(frames[i][1]):
                self.frames.append(frames[i][0])
        self.total_length = len(self.frames)
    def update(self):
        if self.render_mode == RenderMode.LOOP:
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
        elif self.render_mode == RenderMode.ONCE:
            self.current_frame_idx += 1
            if self.current_frame_idx >= len(self.key_frames):
                self.finished = True
                self.current_frame_idx = len(self.key_frames) - 1
        else:
            # exception: invalid render mode
            raise ValueError("Invalid render mode. The render_mode must be a renderMode enum.")
    
    def get_current_frame(self):
        if(self.finished):
            raise ValueError("The sprite has finished playing.")
        original =  self.frames[self.current_frame_idx]
        # transpose x and y to match the format of Pygame
        return original.transpose(1, 0, 2)
    def get_current_frame_idx(self):
        return self.current_frame_idx