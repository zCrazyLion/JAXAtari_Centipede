import pygame
import ale_py
import numpy as np
import os

# Initialize ALE
ale = ale_py.ALEInterface()

# Set the path to your Boxing ROM file
rom_path = "./testRun/Boxing.bin"
ale.loadROM(rom_path)

# Create a directory for screenshots
screenshot_dir = "./frames_boxings"
os.makedirs(screenshot_dir, exist_ok=True)

# Counter for frame filenames
frame_counter = 1
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

def get_current_render():
    """Return the current rendering pixel array."""
    return ale.getScreenRGB()

def save_frame_as_numpy(frame):
    """Save the given frame as a NumPy array with an auto-incremented filename."""
    global frame_counter
    filepath = os.path.join(screenshot_dir, f"frame_{frame_counter}.npy")
    np.save(filepath, frame)
    print(f"Frame saved as NumPy array: {filepath}")
    frame_counter += 1

def main():
    pygame.init()
    screen_width, screen_height = 160 * 4, 210 * 4
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Boxing Game")
    clock = pygame.time.Clock()

    ale.reset_game()

    running = True
    while running:
        frame = get_current_render()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        action = 0  # Default action is no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Save screenshot
                    save_frame_as_numpy(frame)
                else:
                    action = get_action_from_key(event.key)

        ale.act(action)

        if ale.game_over():
            print("Game over!")
            ale.reset_game()

        clock.tick(5)  # Limit to 30 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
