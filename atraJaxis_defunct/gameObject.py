class GameObject:
    """
    A class to represent a game object.

    Attributes
    ----------
    x : int
        The x-coordinate of the game object.
    y : int
        The y-coordinate of the game object.
    sprite : object
        The sprite associated with the game object.
    destroyed : bool
        A flag indicating whether the game object is destroyed.

    Methods
    -------
    move(dx, dy):
        Moves the game object by a given offset.
    displace(x, y):
        Sets the game object to a specific position.
    isDestroyed():
        Checks if the game object is destroyed.
    update():
        Updates the game object's state by updating its sprite and checking its destruction status.
    render(grid):
        Renders the game object onto a given grid.
    """

    def __init__(self, x, y, sprite):
        """
        Initializes the GameObject instance.

        Parameters
        ----------
        x : int
            The x-coordinate of the game object.
        y : int
            The y-coordinate of the game object.
        sprite : object
            The sprite associated with the game object.
        """
        self.x = x
        self.y = y
        self.sprite = sprite
        self.destroyed = False

    def move(self, dx, dy):
        """
        Moves the game object by a given offset.

        Parameters
        ----------
        dx : int
            The change in the x-coordinate.
        dy : int
            The change in the y-coordinate.
        """
        self.x += dx
        self.y += dy

    def displace(self, x, y):
        """
        Sets the game object to a specific position.

        Parameters
        ----------
        x : int
            The new x-coordinate of the game object.
        y : int
            The new y-coordinate of the game object.
        """
        self.x = x
        self.y = y

    def isDestroyed(self):
        """
        Checks if the game object is destroyed.

        Returns
        -------
        bool
            True if the game object is destroyed, False otherwise.
        """
        return self.destroyed

    def update(self):
        """
        Updates the game object's state.

        This method updates the sprite and sets the destroyed flag to True if the sprite is finished.
        """
        self.sprite.update()
        if self.sprite.finished:
            self.destroyed = True

    def render(self, grid):
        """
        Renders the game object onto a given grid.

        Parameters
        ----------
        grid : np.ndarray
            The grid where the game object will be rendered.

        Notes
        -----
        The method places the sprite's pixels onto the grid at the game object's position.
        Pixels outside the grid boundaries are ignored.
        """
        frame = self.sprite.render()
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                # Check if the pixel is within the grid bounds
                if 0 <= i + self.y < grid.shape[0] and 0 <= j + self.x < grid.shape[1]:
                    grid[i + self.y][j + self.x] = frame[i][j]
