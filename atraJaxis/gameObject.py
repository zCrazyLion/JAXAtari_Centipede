DEBUG = True
class gameObject:
    def __init__(self, x, y, sprite):
        self.x = x
        self.y = y
        self.sprite = sprite
        self.destroyed = False

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        
    def isDestroyed(self):
        return self.destroyed
    
    def update(self):
        self.sprite.update()
        if self.sprite.finished:
            self.destroyed = True
        
    def render(self, grid):
        # put the sprite on the layer on the position (x, y)
        frame = self.sprite.get_current_frame()
        idx = self.sprite.get_current_frame_idx()
        if DEBUG:
            print("current frame index: ", idx)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                grid[self.y + i][self.x + j] = frame[i][j]