from .gameObject import GameObject
import numpy as np

class textHUD(GameObject):
    def __init__(self, text, x, y, charToFrame, spaceBetweenChars):
        super().__init__(x, y, None)
        self.text = text
        self.frame = None
        self.charToFrame = charToFrame
        self.spaceBetweenChars = spaceBetweenChars
        self.update()
        
    def render(self, grid):
        frame_t = self.frame.transpose(1, 0, 2)
        # put self.frame on the grid at position (self.x, self.y)
        for i in range(self.width):
            for j in range(self.height):
                grid[self.y + i][self.x + j] = frame_t[i][j]
                
    def update(self):
        self.width = 0
        for char in self.text:
            self.width += self.charToFrame[char].shape[1] + self.spaceBetweenChars
            
        # Then, compute the height of the text: height = maximum of the heights of the characters
        self.height = 0
        for char in self.text:
            self.height = max(self.height, self.charToFrame[char].shape[0])
        
        # Finally, compute the frame of the text
        self.frame = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        currentX = 0
        for char in self.text:
            charFrame = self.charToFrame[char]
            for i in range(charFrame.shape[0]):
                for j in range(charFrame.shape[1]):
                    self.frame[i][currentX + j] = charFrame[i][j]
            currentX += charFrame.shape[1] + self.spaceBetweenChars
        return 
