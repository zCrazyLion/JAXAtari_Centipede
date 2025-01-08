import spriteLoader
from sprite import renderMode
from canvas import canvas
from layer import layer
from gameObject import gameObject
import pygame

sl = spriteLoader.spriteLoader()
sl.loadFrame('./atraJaxis/test_frames/1.npy', name='sub1')
sl.loadFrame('./atraJaxis/test_frames/2.npy', name='sub2')
sl.loadFrame('./atraJaxis/test_frames/3.npy', name='sub3')

sl.loadSprite('player_sub', [('sub1', 4), ('sub2', 4), ('sub3', 4)], renderMode.LOOP)

sub = gameObject(0, 0, sl.getSprite('player_sub'))

windows_width = 100
windows_height = 100

canvas1 = canvas(windows_width, windows_height)
canvas1.addLayer(layer('player_sub', windows_width, windows_height))
canvas1.layers[0].addGameObject(sub)

scaling_factor = 3

# make a pygame window 600 x 400 pixels
pygame.init()
win = pygame.display.set_mode((windows_height*scaling_factor, windows_height*scaling_factor))
clock = pygame.time.Clock()
# main loop
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    grid = canvas1.render()

    
    canvas1.update()
    
    frame_surface = pygame.surfarray.make_surface(grid)
    frame_surface = pygame.transform.scale(frame_surface, (windows_width*scaling_factor, windows_height*scaling_factor))
    win.blit(frame_surface, (0, 0))
    pygame.display.flip()
    clock.tick(3)  # Limit to 60 FPS