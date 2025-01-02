import pygame
import ale_py
import numpy as np
import time

# Initialize ALE
ale = ale_py.ALEInterface()

# Set the path to your Boxing ROM file
rom_path = "testRun\Boxing.bin"
ale.loadROM(rom_path)

def get_action_from_key(key):
    """Map keyboard keys to ALE actions."""
    key_action_map = {
        pygame.K_UP: 2,      # Move Up
        pygame.K_DOWN: 5,    # Move Down
        pygame.K_LEFT: 4,    # Move Left
        pygame.K_RIGHT: 3,   # Move Right
        pygame.K_SPACE: 1    # Punch
    }
    return key_action_map.get(key, 0)  # Default to no-op

def main():
    pygame.init()
    screen_width, screen_height = 160 * 4, 210 * 4
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Boxing Game")
    clock = pygame.time.Clock()

    ale.reset_game()

    running = True
    while running:
        frame = ale.getScreenRGB()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        action = 0  # Default action is no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                action = get_action_from_key(event.key)

        ale.act(action)

        if ale.game_over():
            print("Game over!")
            ale.reset_game()

        clock.tick(30)  # Limit to 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
