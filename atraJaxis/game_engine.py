from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import pygame
from jax import lax
class AgnosticPath(Path): # https://stackoverflow.com/questions/60291545/converting-windows-path-to-linux
    """A class that can handle input with Windows (\\) and/or posix (/) separators for paths"""
    def __new__(cls, *args, **kwargs):
        new_path = PureWindowsPath(*args).parts
        if (os.name != "nt") and (len(new_path) > 0) and (new_path[0] in ("/", "\\")):
          new_path = ("/", *new_path[1:])
        return super().__new__(Path, *new_path, **kwargs)

def loadFrame(fileName):
    # Load frame (np array) from a .npy file and convert to jnp array
    frame = jnp.load(AgnosticPath(fileName))
    # Check if the frame's shape is [[[r, g, b, a], ...], ...]
    if frame.ndim != 3 or frame.shape[2] != 4:
        raise ValueError("Invalid frame format. The frame must have a shape of (height, width, 4).")
    return frame

@partial(jax.jit, static_argnames=["flip_horizontal", "flip_vertical"])
def flipSprite(sprite, flip_horizontal=False, flip_vertical=False):
    # Use jnp.transpose for JAX compatibility
    transposed = jnp.transpose(sprite, (1, 0, 2))  # Transpose x and y axes

    if flip_horizontal:
        return transposed[:, ::-1, :]
    elif flip_vertical:
        return transposed[::-1, :, :]
    else:
        return transposed



def get_sprite_frame(frames, frame_idx, flip_horizontal=False, flip_vertical=False, loop=True):
    frame_idx_converted = jax.lax.cond(loop, lambda x: x % len(frames), lambda x: x, frame_idx)
    if frame_idx_converted < 0 or frame_idx_converted >= len(frames):
        return jnp.zeros((1,1,4))  # Return a blank frame if the index is out of bounds
    original = frames[frame_idx_converted] # get the frame as a jnp array
    rendered = flipSprite(original, flip_horizontal, flip_vertical)
    # Apply vertical flip
    return rendered

@jax.jit
def render_at(raster, x, y, sprite, destroyed=False):
    if destroyed:
        return raster
    # Get the dimensions of the sprite
    sprite_height, sprite_width, _ = sprite.shape
    raster_height, raster_width, _ = raster.shape

    # Ensure x and y are within valid range
    x = jnp.clip(x, 0, raster_width - sprite_width)
    y = jnp.clip(y, 0, raster_height - sprite_height)

    # Dynamically extract the raster crop
    raster_crop = lax.dynamic_slice(raster, (y, x, 0), (sprite_height, sprite_width, 3))
    sprite_crop = sprite[:, :, :3]  # Always take full sprite RGB channels
    alpha = sprite[:, :, 3:] / 255  # Alpha channel normalization

    # Alpha blending (this should correctly handle transparency)
    blended_crop = sprite_crop * (alpha) + raster_crop * (1 - alpha)

    # Dynamically update raster with blended crop
    new_raster = lax.dynamic_update_slice(raster, blended_crop, (y, x, 0))

    return new_raster

def update_pygame(pygame_screen, raster, SCALING_FACTOR=3, WIDTH=400, HEIGHT=300):
    pygame_screen.fill((0, 0, 0))

    # Convert JAX array to NumPy and ensure uint8 format
    raster = np.array(raster)  # Convert from JAX to NumPy
    raster = raster.astype(np.uint8) # Ensure uint8 format

    
    # Convert to Pygame surface and scale to screen size
    frame_surface = pygame.surfarray.make_surface(raster)
    frame_surface = pygame.transform.scale(frame_surface, (WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))

    pygame_screen.blit(frame_surface, (0, 0))
    pygame.display.flip()




    
if __name__ == "__main__":

    sub1 = loadFrame("./atraJaxis/test_frames/1.npy")
    sub2 = loadFrame("./atraJaxis/test_frames/2.npy")
    sub3 = loadFrame("./atraJaxis/test_frames/3.npy")
    
    shark1 = loadFrame("./sprites/seaquest/shark/1.npy")
    shark2 = loadFrame("./sprites/seaquest/shark/2.npy")

    # render an 2d RGBA array in pygame and update at a frame rate of 60fps
    pygame.init()
    
    SCALING_FACTOR = 3
    WIDTH = 400
    HEIGHT = 300

    screen = pygame.display.set_mode((WIDTH * SCALING_FACTOR, HEIGHT * SCALING_FACTOR))
    clock = pygame.time.Clock()
    running = True
    frame_idx = 0
    # establish a frame of WIDTH x HEIGHT
    empty_frame = np.zeros((WIDTH, HEIGHT, 3))
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # render the frame
        raster = empty_frame
        # render the 1st frame at (0, 0)
        sub_sprite = [sub1,sub1,sub1,sub1,sub2,sub2,sub2,sub2,sub3,sub3, sub3,sub3]
        sub_frame = get_sprite_frame(sub_sprite, frame_idx, loop=False)
        shark_sprite = [shark1, shark1, shark2, shark2]
        shark_frame = get_sprite_frame(shark_sprite, frame_idx, loop=True)
        raster = render_at(raster, 140, 140, sub_frame)
        raster = render_at(raster, 100, 100, shark_frame)
        update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        frame_idx += 1
        clock.tick(60)
    pygame.quit()
