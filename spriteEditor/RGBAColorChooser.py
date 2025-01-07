import tkinter as tk
from tkinter.colorchooser import askcolor

class RGBAColorChooser(tk.Toplevel):
    def __init__(self, master, initial_color=[0, 0, 0, 255]):
        super().__init__(master)
        self.title("RGBA Color Picker")
        self.geometry("300x150")
        self.initial_color = initial_color

        # Create a frame for the color input controls
        self.color_frame = tk.Frame(self)
        self.color_frame.pack(pady=10)

        # RGB color picker button
        self.rgb_button = tk.Button(self.color_frame, text="Pick RGB", command=self.open_rgb_picker)
        self.rgb_button.grid(row=0, column=0, padx=5, pady=5)

        # Alpha input
        self.alpha_label = tk.Label(self.color_frame, text="Alpha (0-255):")
        self.alpha_label.grid(row=1, column=0, padx=5, pady=5)

        self.alpha_input = tk.Entry(self.color_frame)
        self.alpha_input.insert(0, str(self.initial_color[3]))  # Set initial alpha value
        self.alpha_input.grid(row=1, column=1, padx=5, pady=5)

        # OK and Cancel buttons
        self.buttons_frame = tk.Frame(self)
        self.buttons_frame.pack(pady=10)

        self.ok_button = tk.Button(self.buttons_frame, text="OK", command=self.on_ok)
        self.ok_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = tk.Button(self.buttons_frame, text="Cancel", command=self.on_cancel)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Variable to store selected RGBA color
        self.selected_rgba = None

    def open_rgb_picker(self):
        # Open the RGB color picker
        rgb, _ = askcolor(color=self.rgba_to_hex(self.initial_color), parent=self)
        if rgb:
            # Update initial RGB color
            self.initial_color = [int(rgb[0]), int(rgb[1]), int(rgb[2]), self.initial_color[3]]
            self.alpha_input.delete(0, tk.END)
            self.alpha_input.insert(0, str(self.initial_color[3]))  # Preserve current alpha

    def on_ok(self):
        # Get the alpha value from input
        try:
            alpha = int(self.alpha_input.get())
            if alpha < 0 or alpha > 255:
                raise ValueError("Alpha value must be between 0 and 255.")
        except ValueError as e:
            tk.messagebox.showerror("Invalid Input", str(e))
            return
        
        # Update the selected RGBA color
        self.selected_rgba = self.initial_color[:3] + [alpha]
        self.destroy()

    def on_cancel(self):
        # If cancelled, do not change the color
        self.selected_rgba = self.initial_color
        self.destroy()

    def rgba_to_hex(self, rgba):
        # Convert RGBA to hex
        rgba = [int(c) for c in rgba]
        return f"#{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}"

