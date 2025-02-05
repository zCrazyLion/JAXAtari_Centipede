from pathlib import Path, PureWindowsPath
import os
import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
import pygame
from jax import lax


class AgnosticPath(Path):
    """A class that can handle input with Windows (\\) and/or posix (/) separators for paths"""

    def __new__(cls, *args, **kwargs):
        # Convert the input to a PureWindowsPath first to normalize backslashes
        win_path = PureWindowsPath(*args)
        # Get the parts of the path
        parts = win_path.parts

        # Handle root paths differently on non-Windows systems
        if os.name != "nt" and len(parts) > 0:
            # If the path started with a drive letter (e.g., 'C:'), remove it
            if len(parts[0]) == 2 and parts[0][1] == ':':
                parts = parts[1:]
            # If it's an absolute path, ensure it starts with '/'
            if parts and not parts[0] in ('/', '\\'):
                parts = ('/',) + parts

        # Use the superclass's __new__ to create the Path object
        return super().__new__(cls, *parts, **kwargs)

def loadFrame(fileName):
    # Load frame (np array) from a .npy file and convert to jnp array
    frame = jnp.load(fileName)
    # Check if the frame's shape is [[[r, g, b, a], ...], ...]
    if frame.ndim != 3 or frame.shape[2] != 4:
        raise ValueError("Invalid frame format. The frame must have a shape of (height, width, 4).")
    if frame.shape[0] > frame.shape[1]:  # Height > Width indicates wrong orientation
        frame = jnp.transpose(frame, (1, 0, 2))
    return frame

@partial(jax.jit, static_argnames=["flip_horizontal", "flip_vertical"])
def flipSprite(sprite, flip_horizontal=False, flip_vertical=False):
    # Use jnp.transpose for JAX compatibility
    transposed = jnp.transpose(sprite, (1, 0, 2))  # Transpose x and y axes

    if flip_vertical:
        return transposed[:, ::-1, :]
    elif flip_horizontal:
        return transposed[::-1, :, :]
    else:
        return transposed


@jax.jit
def get_sprite_frame(frames, frame_idx, loop=True):
    num_frames = frames.shape[0]

    # Handle looping
    frame_idx_looped = jnp.mod(frame_idx, num_frames)
    frame_idx_converted = jax.lax.cond(loop,
                                      lambda _: frame_idx_looped,
                                      lambda _: frame_idx,
                                      operand=None)

    # Create bounds check using jax.lax.cond
    valid_frame = jnp.logical_and(
        frame_idx_converted >= 0,
        frame_idx_converted < num_frames
    )

    # Get frame dimensions and create blank frame with matching dtype
    frame_height = frames.shape[1]
    frame_width = frames.shape[2]
    frame_channels = frames.shape[3]
    blank_frame = jnp.zeros((frame_height, frame_width, frame_channels), dtype=frames.dtype)

    # Return either the frame or blank frame based on validity check
    return jax.lax.cond(
        valid_frame,
        lambda _: frames[frame_idx_converted],  # removed unnecessary jnp.array() call
        lambda _: blank_frame,
        operand=None
    )


@partial(jax.jit)
def render_at(raster, y, x, sprite_frame, flip_horizontal=False, flip_vertical=False):
    """Renders a sprite onto a raster at position (x,y) with optional flipping.

    Args:
        raster: JAX array of shape (width, height, 3) for the target image
        y: Integer y coordinate for sprite placement
        x: Integer x coordinate for sprite placement
        sprite_frame: JAX array of shape (height, width, 4) containing RGB + alpha
    """
    # Get dimensions correctly - sprite is in (height, width) format
    sprite_height, sprite_width, _ = sprite_frame.shape
    raster_width, raster_height, _ = raster.shape

    # Clip coordinates
    x = jnp.clip(x, 0, raster_width - sprite_width)
    y = jnp.clip(y, 0, raster_height - sprite_height)

    # Create sprite array and handle flipping - axis 0 is height, axis 1 is width
    sprite = jnp.array(sprite_frame)
    sprite = jnp.where(
        flip_horizontal,
        jnp.flip(sprite, axis=1),  # Flip width dimension
        sprite
    )
    sprite = jnp.where(
        flip_vertical,
        jnp.flip(sprite, axis=0),  # Flip height dimension
        sprite
    )

    # Rest remains same but with corrected dimensions
    sprite_rgb = sprite[..., :3]
    alpha = sprite[..., 3:] / 255.0

    # Use correct dimension ordering in slicing
    raster_region = jax.lax.dynamic_slice(
        raster,
        (x.astype(int), y.astype(int), 0),
        (sprite_height, sprite_width, 3)  # Note width, height order to match raster
    )

    blended = sprite_rgb * alpha + raster_region * (1.0 - alpha)

    new_raster = jax.lax.dynamic_update_slice(
        raster,
        blended,
        (x.astype(int), y.astype(int), 0)
    )

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

    sub1 = loadFrame("./sprites/seaquest/player_sub/1.npy")
    sub2 = loadFrame("./sprites/seaquest/player_sub/2.npy")
    sub3 = loadFrame("./sprites/seaquest/player_sub/3.npy")
    sub_sprite = [sub1,sub1,sub1,sub1,sub2,sub2,sub2,sub2,sub3,sub3, sub3,sub3]

    shark1 = loadFrame("./sprites/seaquest/shark/1.npy")
    shark2 = loadFrame("./sprites/seaquest/shark/2.npy")
    shark_sprite = [shark1, shark1, shark2, shark2]
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


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # render the frame
        raster = jnp.zeros((WIDTH, HEIGHT, 3))
        # render the 1st frame at (0, 0)
        sub_frame = get_sprite_frame(sub_sprite, frame_idx, loop=True)
        shark_frame = get_sprite_frame(shark_sprite, frame_idx, loop=True)
        raster = render_at(raster, 140, 140, sub_frame, flip_horizontal=True)
        raster = render_at(raster, 100, 100, shark_frame)
        update_pygame(screen, raster, SCALING_FACTOR, WIDTH, HEIGHT)
        frame_idx += 1
        clock.tick(60)
    pygame.quit()
