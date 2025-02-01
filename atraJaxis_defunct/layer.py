import numpy as np

class Layer:
    """
    A class to represent a layer in a game.

    Attributes
    ----------
    name : str
        The name of the layer.
    gameObjects : list
        A list to store game objects in the layer.
    width : int
        The width of the layer.
    height : int
        The height of the layer.

    Methods
    -------
    addGameObject(gameObject):
        Adds a game object to the layer. This only adds the reference to the game object, so the game object should be created before adding it to the layer.
    removeGameObject(gameObject):
        Removes a game object from the layer.
    update():
        Updates the state of the layer by removing destroyed game objects and updating the remaining ones.
    render():
        Renders the layer and returns a grid representation of the layer. The grid has a shape of (height, width, 4). The first three channels represent the RGB values of the pixels, and the last channel represents the alpha values of the pixels.
    """

    def __init__(self, name, width, height):
        """
        Initializes the Layer object.

        Parameters
        ----------
        name : str
            The name of the layer.
        width : int
            The width of the layer.
        height : int
            The height of the layer.
        """
        self.name = name
        self.gameObjects = []
        self.width = width
        self.height = height

    def addGameObject(self, gameObject):
        """
        Adds a game object to the layer.

        Parameters
        ----------
        gameObject : object
            The game object to be added.
        """
        self.gameObjects.append(gameObject)

    def removeGameObject(self, gameObject):
        """
        Removes a game object from the layer.

        Parameters
        ----------
        gameObject : object
            The game object to be removed.
        """
        self.gameObjects.remove(gameObject)

    def update(self):
        """
        Updates the state of the layer.

        This method removes destroyed game objects and updates the remaining ones.
        """
        # Remove destroyed gameObjects
        self.gameObjects = [gameObject for gameObject in self.gameObjects if not gameObject.isDestroyed()]

        # Update remaining gameObjects
        for gameObject in self.gameObjects:
            gameObject.update()

    def render(self):
        """
        Renders the layer.

        This method generates a grid representation of the layer, with the first three channels representing RGB values
        and the fourth channel representing alpha values.

        Returns
        -------
        np.ndarray
            A grid representation of the layer with shape (height, width, 4).
        """
        grid = np.zeros((self.height, self.width, 4))
        for gameObject in self.gameObjects:
            gameObject.render(grid)
        return grid
