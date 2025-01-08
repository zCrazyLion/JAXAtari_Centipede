# class of a layer.
# A layer a collection of gameObjects.
import numpy as np
class layer:
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
        Adds a game object to the layer.
    removeGameObject(gameObject):
        Removes a game object from the layer.
    update():
        Updates the state of the layer by removing destroyed game objects and updating the remaining ones.
    render():
        Renders the layer and returns a grid representation of the layer. The grid has a shape of (height, width, 4). The first three channels represent the RGB values of the pixels, and the last channel represents the alpha values of the pixels.
    """
    def __init__(self, name, width, height):
        self.name = name
        self.gameObjects = []
        self.width = width
        self.height = height
        
    def addGameObject(self, gameObject):
        self.gameObjects.append(gameObject)
    def removeGameObject(self, gameObject):
        self.gameObjects.remove(gameObject)
    def update(self):
        # remove destroyed gameObjects
        self.gameObjects = [gameObject for gameObject in self.gameObjects if not gameObject.isDestroyed()]
        
        # update gameObjects
        for gameObject in self.gameObjects:
            gameObject.update()
            
    
    def render(self):
        grid = np.zeros((self.height, self.width))
        for gameObject in self.gameObjects:
            gameObject.render(grid)
        return grid
        