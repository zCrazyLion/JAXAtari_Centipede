from .gameObject import GameObject
import numpy as np

class TextHUD(GameObject):
    """
    A class to represent a text-based Heads-Up Display (HUD) in a game.
    Attributes:
    -----------
    text : str
        The text to be displayed on the HUD.
    x : int
        The x-coordinate of the HUD's position.
    y : int
        The y-coordinate of the HUD's position.
    charToFrame : dict
        A dictionary mapping characters to their corresponding frame representations.
    spaceBetweenChars : int
        The space between characters in the HUD.
    frame : np.ndarray
        The frame representing the entire text.
    width : int
        The width of the text frame.
    height : int
        The height of the text frame.
    Methods:
    --------
    __init__(text, x, y, charToFrame, spaceBetweenChars):
        Initializes the TextHUD with the given text, position, character frames, and space between characters.
    render(grid):
        Renders the text frame onto the given grid at the HUD's position.
    update():
        Updates the text frame based on the current text and character frames.
    """
    def __init__(self, text, x, y, charToFrame, spaceBetweenChars):
        super().__init__(x, y, None)
        self.text = text
        self.frame = None
        self.charToFrame = charToFrame
        self.spaceBetweenChars = spaceBetweenChars
        self.update()
        
    def render(self, grid):
        """
        Renders the frame onto the provided grid at the specified position.

        Args:
            grid (list of list of any): The grid where the frame will be rendered.

        The frame is transposed before being placed on the grid. The position where
        the frame is placed is determined by the attributes `self.x` and `self.y`.
        The frame is placed only within the bounds of the grid.
        """
        frame_t = self.frame.transpose(1, 0, 2)
        # put self.frame on the grid at position (self.x, self.y)
        for i in range(self.width):
            for j in range(self.height):
                if self.x + j < len(grid[0]) and self.y + i < len(grid) and self.x + j >= 0 and self.y + i >= 0:
                    grid[self.y + i][self.x + j] = frame_t[i][j]

                
    def update(self):
        """
        Updates the dimensions and frame of the text based on the characters and their spacing.
        This method performs the following steps:
        1. Computes the total width of the text by summing the widths of each character frame and the space between characters.
        2. Computes the height of the text as the maximum height among all character frames.
        3. Creates a new frame (numpy array) with the computed dimensions and fills it with the character frames.
        Attributes:
            width (int): The total width of the text.
            height (int): The maximum height of the text.
            frame (np.ndarray): The frame representing the text.
            currentX (int): The current x-coordinate for placing the next character frame.
        """
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

class BarHUD(GameObject):
    def __init__(self, x, y, width, height, max_value, current_value, color):
        """
        Initializes a new HUD element.

        Args:
            x (int): The x-coordinate of the HUD element.
            y (int): The y-coordinate of the HUD element.
            width (int): The width of the HUD element.
            height (int): The height of the HUD element.
            max_value (int): The maximum value for the HUD element.
            current_value (int): The current value of the HUD element.
            color (tuple): The color of the HUD element, typically in RGB format.
        """
        super().__init__(x, y, None)
        self.width = width
        self.height = height
        self.max_value = max_value
        self.current_value = current_value
        self.color = color
        self.frame = None
        self.update()
        
    def render(self, grid):
        """
        Renders the frame onto the provided grid at the position (self.x, self.y).

        Args:
            grid (list of list of any): The grid where the frame will be rendered.

        Notes:
            - The frame is rendered only within the bounds of the grid.
            - The frame is a 2D list stored in self.frame.
            - The position (self.x, self.y) specifies the top-left corner of the frame on the grid.
        """
        # put self.frame on the grid at position (self.x, self.y)
        for i in range(self.width):
            for j in range(self.height):
                if self.y + i < len(grid) and self.x + j < len(grid[0]) and self.x + j >= 0 and self.y + i >= 0:
                    grid[self.y + i][self.x + j] = self.frame[i][j]
                    
    def update(self):
        """
        Updates the frame with the current value of the bar.

        This method initializes the frame as a zero-filled numpy array with the shape
        (width, height, 4). It then calculates the current width of the bar based on
        the ratio of current_value to max_value. The bar is filled with the specified
        color up to the calculated width.

        Attributes:
            frame (np.ndarray): The frame to be updated.
            current_value (int): The current value of the bar.
            max_value (int): The maximum value of the bar.
            width (int): The width of the frame.
            height (int): The height of the frame.
            color (tuple): The color to fill the bar with.
        """
        self.frame = np.zeros((self.width, self.height, 4), dtype=np.uint8)
        # Compute the current width of the bar
        current_width = int(self.current_value / self.max_value * self.width)
        # Fill the bar with the color
        for i in range(current_width):
            for j in range(self.height):
                self.frame[i][j] = self.color
        return