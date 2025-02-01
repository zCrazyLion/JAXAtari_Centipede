import numpy as np

class Canvas:
    """
    A class that represents a canvas for rendering layers.

    Attributes:
        width (int): The width of the canvas.
        height (int): The height of the canvas.
        layers (list): A list of layers added to the canvas.
    """

    def __init__(self, width, height):
        """
        Initializes the Canvas object.

        Parameters:
            width (int): The width of the canvas.
            height (int): The height of the canvas.
        """
        self.layers = []
        self.width = width
        self.height = height

    def addLayer(self, layer):
        """
        Adds a new layer to the canvas.

        Parameters:
            layer (Layer): The layer to add. The layer's dimensions must match the canvas dimensions.

        Raises:
            ValueError: If the layer's dimensions do not match the canvas dimensions.
        """
        if layer.width != self.width or layer.height != self.height:
            raise ValueError("The layer size must match the canvas size.")
        self.layers.append(layer)

    def removeLayer(self, layer):
        """
        Removes a layer from the canvas.

        Parameters:
            layer (Layer): The layer to remove.

        Raises:
            ValueError: If the layer is not found in the canvas.
        """
        self.layers.remove(layer)

    def getLayer(self, name):
        """
        Retrieves a layer by its name.

        Parameters:
            name (str): The name of the layer to retrieve.

        Returns:
            Layer: The layer with the specified name.

        Raises:
            ValueError: If no layer with the specified name is found.
        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise ValueError(f"Layer {name} not found.")

    def update(self):
        """
        Updates all layers in the canvas.
        """
        for layer in self.layers:
            layer.update()

    def render(self):
        """
        Renders the canvas by blending all layers sequentially.

        Returns:
            np.ndarray: The rendered RGB image of the canvas.
        """
        grid = np.zeros((self.height, self.width, 3))
        for layer in self.layers:
            layer_grid = layer.render()
            grid = self.blend(grid, layer_grid)
        return grid

    def blend(self, base, newLayer):
        """
        Blends an RGBA layer on top of an RGB base image using alpha blending.

        Parameters:
            base (np.ndarray): Base image of shape (height, width, 3).
            newLayer (np.ndarray): New layer of shape (height, width, 4), with an alpha channel.

        Returns:
            np.ndarray: Blended image of shape (height, width, 3).

        Raises:
            ValueError: If the dimensions of base and newLayer do not match.
        """
        # Extract the RGB channels and the alpha channel from the new layer
        newLayerRGB = newLayer[..., :3]  # Shape: (height, width, 3)
        alpha = newLayer[..., 3] / 255.0  # Normalize alpha to [0, 1], Shape: (height, width)

        # Ensure base and newLayer dimensions match
        if base.shape[:2] != newLayer.shape[:2]:
            raise ValueError("Base and newLayer dimensions do not match.")

        # Blend the two images using the alpha channel
        blended = (1 - alpha)[..., None] * base + alpha[..., None] * newLayerRGB

        # Ensure the output values are in the valid range for RGB
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended
