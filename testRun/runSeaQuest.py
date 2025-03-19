import pygame
import ale_py
import numpy as np
import os

# Initialize ALE
ale = ale_py.ALEInterface()

# Set the path to your ROM file
rom_path = "./testRun/seaquest.bin"
ale.loadROM(rom_path)

# Create a directory for screenshots (stored as NumPy arrays)
screenshot_dir = "frames_numpy"
os.makedirs(screenshot_dir, exist_ok=True)

# Counter for frame filenames
frame_counter = 1


def get_action_from_key(key):
    """Map keyboard keys to ALE actions."""
    key_action_map = {
        pygame.K_UP: 2,  # Move Up
        pygame.K_DOWN: 5,  # Move Down
        pygame.K_LEFT: 4,  # Move Left
        pygame.K_RIGHT: 3,  # Move Right
        pygame.K_SPACE: 1,  # Fire torpedoes
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


# Add a global set to track pressed keys
pressed_keys = set()


def main():
    pygame.init()
    screen_width, screen_height = 160 * 4, 210 * 4
    screen = pygame.display.set_mode((screen_width, screen_height))
    base_caption = "Seaquest Game"
    pygame.display.set_caption(base_caption)
    clock = pygame.time.Clock()

    ale.reset_game()

    running = True
    paused = False

    while running:
        frame = get_current_render()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame_surface = pygame.transform.scale(
            frame_surface, (screen_width, screen_height)
        )
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        action = 0  # Default action is no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                pressed_keys.add(event.key)
                if event.key == pygame.K_s:  # Save frame as NumPy array
                    save_frame_as_numpy(frame)
                elif event.key == pygame.K_p:  # Toggle pause
                    paused = not paused
                    pygame.display.set_caption(
                        f"{base_caption} (Paused)" if paused else base_caption
                    )
                elif paused and event.key == pygame.K_f:  # Step one frame
                    action = ale.act(action)  # Execute one frame of the game
                    print("Stepped one frame.")
                else:
                    action = get_action_from_key(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in pressed_keys:
                    pressed_keys.remove(event.key)

        if not paused:
            # Determine the action based on pressed keys
            for key in pressed_keys:
                action = get_action_from_key(key)
                if action != 0:  # If any action is mapped, use it
                    break

            ale.act(action)

        if ale.game_over():
            print("Game over!")
            ale.reset_game()

        clock.tick(60)  # Limit to 60 FPS

    pygame.quit()


if __name__ == "__main__":
    main()
