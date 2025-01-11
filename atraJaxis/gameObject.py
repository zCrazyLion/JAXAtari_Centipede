class GameObject:
    def __init__(self, x, y, sprite):
        self.x = x
        self.y = y
        self.sprite = sprite
        self.destroyed = False

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        
    def displace(self, x, y):
        self.x = x
        self.y = y 
        
    def isDestroyed(self):
        return self.destroyed
    
    def update(self):
        self.sprite.update()
        if self.sprite.finished:
            self.destroyed = True
        
    def render(self, grid):
        # put the sprite on the layer on the position (x, y)
        frame = self.sprite.render()
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                # if the pixel is in the range of the grid, put the pixel on the grid
                if 0 <= i + self.y < grid.shape[0] and 0 <= j + self.x < grid.shape[1]:
                    grid[i + self.y][j + self.x] = frame[i][j]