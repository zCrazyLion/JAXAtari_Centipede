# A canvas is a collection of layers.
# The layers are rendered in order, with the first layer rendered first.
# Render each layer on top of the previous layer with alpha blending. The layers are rendered as RGB images.
import numpy as np
class canvas:
    def __init__(self, width, height):
        self.layers = []
        self.width = width
        self.height = height
    def addLayer(self, layer):
        if layer.width != self.width or layer.height != self.height:
            raise ValueError("The layer size must match the canvas size.")
        self.layers.append(layer)
    def removeLayer(self, layer):
        self.layers.remove(layer)
    def update(self):
        for layer in self.layers:
            layer.update()
    def render(self):
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
            newLayer (np.ndarray): New layer of shape (height, width, 4), with alpha channel.

        Returns:
            np.ndarray: Blended image of shape (height, width, 3).
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