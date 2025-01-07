import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.colorchooser import askcolor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from PIL import Image, ImageTk

class NPYImageEditor:
    def __init__(self, master):
        self.root = master
        self.root.title("NPY Image Editor")

        self.image = None
        self.zoom_level = 1
        self.current_color = [0, 0, 0]  # Default color: black
        self.mouse_pressed = False
        self.create_widgets()
        self.tool = None  # Initialize tool attribute
        self.selected = None
        self.selection_start = None
        self.selection_end = None
        self.state_queue = []
        self.current_state_index = -1
        
        # Bind keyboard shortcuts
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-a>", self.select_all)
        self.root.bind("<Control-d>", self.deselect_all)
        self.root.bind("<Control-s>", self.save_selection)


        # State Queue UI with Scrollbar
        self.state_queue_frame = tk.Frame(self.root)
        self.state_queue_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # Add a canvas for the state queue with an associated scrollbar
        self.state_canvas = tk.Canvas(self.state_queue_frame, width=200, height=300)
        self.state_canvas.pack(side=tk.LEFT, padx=5, fill=tk.Y)

        self.scrollbar = tk.Scrollbar(self.state_queue_frame, orient=tk.VERTICAL, command=self.state_canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.state_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.image_thumbnails = []  # List to store thumbnails for each state

    def create_widgets(self):
        # Menu
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save Selection", command=self.save_selection)
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)

        # Canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)

        # Tool Buttons
        tools_frame = tk.Frame(self.root)
        tools_frame.pack(fill=tk.X)

        tk.Button(tools_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT)
        tk.Button(tools_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT)
        tk.Button(tools_frame, text="Pencil", command=self.activate_pencil).pack(side=tk.LEFT)
        tk.Button(tools_frame, text="Magic Wand", command=self.activate_magic_wand).pack(side=tk.LEFT)
        tk.Button(tools_frame, text="Rectangular Selection", command=self.activate_rectangular_selection).pack(side=tk.LEFT)
        tk.Button(tools_frame, text="Dropper", command=self.activate_dropper).pack(side=tk.LEFT)

        # Current Color Indicator
        self.color_indicator = tk.Label(tools_frame, text="", bg=self.rgb_to_hex(self.current_color), width=5, relief=tk.RAISED)
        self.color_indicator.pack(side=tk.LEFT, padx=5)
        self.color_indicator.bind("<Button-1>", self.open_color_palette)

        self.tool = None

    def update_state(self, last_step_name=None):

        new_state = self.get_current_state(last_step_name)

        if self.current_state_index < len(self.state_queue) - 1:
            self.state_queue = self.state_queue[0:self.current_state_index + 1]

        self.state_queue.append(new_state)

        self.current_state_index = len(self.state_queue) - 1

        self.update_display()
        self.update_state_queue_display()

    def update_state_queue_display(self):
        self.state_canvas.delete("all")  # Clear the canvas before drawing new items

        # Clear the image_thumbnails list before creating new ones
        self.image_thumbnails = []

        y_position = 10  # Start from top position for the first state

        for idx, state in enumerate(self.state_queue):
            image_thumb = self.create_thumbnail(state["image"])
            self.image_thumbnails.append(image_thumb)  # Store the thumbnail

            # Create an image for the thumbnail on the left
            self.state_canvas.create_image(10, y_position, anchor=tk.NW, image=image_thumb)

            # Set the text color (grey for states after the current state)
            text_color = "grey" if idx > self.current_state_index else "black"

            # Add text to the right of the image
            self.state_canvas.create_text(60, y_position + 10, anchor=tk.NW, text=state["last_step_name"], fill=text_color)

            y_position += 60  # Move the next state a bit lower on the canvas

        # Automatically scroll to the bottom after updating
        self.state_canvas.config(scrollregion=self.state_canvas.bbox("all"))
        self.state_canvas.yview_moveto(1)  # Scroll to the bottom
  
        
        # Get the current state of the editor
    def get_current_state(self, last_step_name=None):
        return {
            # clone the reference type
            "image": self.image.copy(),
            "zoom_level": self.zoom_level,
            "current_color": self.current_color.copy(),
            "selected": self.selected.copy(),
            "last_step_name": self.tool if last_step_name is None else last_step_name
        }
    # load the state to the editor
    def load_state(self, state):
        self.image = state["image"].copy()
        self.zoom_level = state["zoom_level"]
        self.current_color = state["current_color"].copy()
        self.selected = state["selected"].copy()
        self.update_display()
        
    # undo the last step
    def undo(self, _):
        if self.current_state_index > 0:
            # Go back one step
            self.current_state_index -= 1
            state_to_load = self.state_queue[self.current_state_index]
            
            # Load the previous state and update the state queue
            self.load_state(state_to_load)
            self.update_state_queue_display()
        else:
            print("No more steps to undo.")

    def redo(self, _):
        if self.current_state_index < len(self.state_queue) - 1:
            # Move forward one step
            self.current_state_index += 1
            state_to_load = self.state_queue[self.current_state_index]
            
            # Load the next state and update the state queue
            self.load_state(state_to_load)
            self.update_state_queue_display()
        else:
            print("No more steps to redo.")

            
            
        # Create a thumbnail of the image
    def create_thumbnail(self, image):
        # Resize the image to create a thumbnail (e.g., 50x50 pixels)
        img_pil = Image.fromarray(image)  # Convert NumPy array to a PIL Image
        img_resized = img_pil.resize((50, 50))  # Resize using interpolation

        # Create a Tkinter PhotoImage from the resized PIL image
        return ImageTk.PhotoImage(img_resized)
        
    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if not file_path:
            return

        try:
            self.image = np.load(file_path)
            if len(self.image.shape) != 3 or self.image.shape[2] != 3:
                raise ValueError("Invalid NPY file format. Expected 3D array with RGB channels.")
            self.zoom_level = 1
            self.update_display()
            self.selected = np.zeros(self.image.shape[:2], dtype=bool)
            self.update_state("open_file")
        except (ValueError, IOError) as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")

    def save_selection(self):
        if self.selection_start and self.selection_end:
            y0, x0 = self.selection_start
            y1, x1 = self.selection_end
            selected_region = self.image[min(y0, y1):max(y0, y1)+1, min(x0, x1):max(x0, x1)+1]

            file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])
            if file_path:
                np.save(file_path, selected_region)
                messagebox.showinfo("Saved", f"Selection saved to {file_path}")
        else:
            messagebox.showwarning("Warning", "No selection to save.")
            
    def select_all(self):
        if self.image is not None:
            self.selected = np.ones(self.image.shape[:2], dtype=bool)  # Select all pixels
            self.update_state("select all")

    def deselect_all(self):
        if self.image is not None:
            self.selected = np.zeros(self.image.shape[:2], dtype=bool)  # Deselect all pixels
            self.update_state("deselect")



    def zoom_in(self):
        self.zoom_level *= 1.5
        self.update_display()

    def zoom_out(self):
        self.zoom_level /= 1.5
        self.update_display()

    def activate_pencil(self):
        self.tool = "pencil"

    def activate_magic_wand(self):
        self.tool = "magic_wand"

    def activate_rectangular_selection(self):
        self.tool = "rectangular_selection"

    def activate_dropper(self):
        self.tool = "dropper"

    def open_color_palette(self, _):
        color = askcolor(color=self.rgb_to_hex(self.current_color), title="Choose Color")
        if color[0]:
            self.current_color = [int(c) for c in color[0]]
            self.color_indicator.config(bg=self.rgb_to_hex(self.current_color))

    def on_mouse_press(self, event):
        if self.image is None or event.xdata is None or event.ydata is None:
            return

        self.mouse_pressed = True 
        x, y = int(event.xdata), int(event.ydata)

        if self.tool == "pencil":
            self.image[y, x] = self.current_color
            self.update_display()
        elif self.tool == "rectangular_selection":
            # reset the selected region
            self.selected = np.zeros(self.image.shape[:2], dtype=bool)
            # mark the starting point of the selection
            self.selection_start = (y, x)
        elif self.tool == "dropper":
            self.current_color = self.image[y, x].tolist()
            self.color_indicator.config(bg=self.rgb_to_hex(self.current_color))

    def on_mouse_release(self, event): # Added to keep track of mouse button state
        
        if self.mouse_pressed and event.xdata and event.ydata:
            if self.tool == "rectangular_selection":
                self.selection_end = (int(event.ydata), int(event.xdata))
                # mark the rectangle between selection_start and selection_end as selected
                y0, x0 = self.selection_start
                y1, x1 = self.selection_end
                self.selected[y0:y1+1, x0:x1+1] = True
                self.update_state("rectangular_selection")
                
            if self.tool == "pencil":
                self.update_state("pencil")
        self.mouse_pressed = False

    def on_mouse_motion(self, event):
        if event.xdata and event.ydata:
            if self.tool == "pencil" and self.mouse_pressed:
                x, y = int(event.xdata), int(event.ydata)
                self.image[y, x] = self.current_color

                
            if self.tool == 'rectangular_selection' and self.mouse_pressed:
                self.selection_end = (int(event.ydata), int(event.xdata))
        
        
        self.update_display()

    def update_display(self):
        if self.image is None:
            return

        self.ax.clear()
        
        # Display the base image
        zoomed_image = self.image[::int(1/self.zoom_level), ::int(1/self.zoom_level)]
        self.ax.imshow(zoomed_image)
        
        # Mark the current rectangular selection with a rectangle border
        if self.selection_start and self.selection_end:
            y0, x0 = self.selection_start
            y1, x1 = self.selection_end
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor="r", facecolor="none")
            self.ax.add_patch(rect)

        # Highlight the selected region without darkening other pixels
        if self.selected is not None and self.selected.any():
            # Create an RGBA overlay image with transparency
            overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
            overlay[self.selected] = [0, 0, 255, 128]  # Blue with 50% opacity
            
            # Display the overlay on top of the image
            self.ax.imshow(overlay, interpolation="none")

        self.canvas.draw()


    def rgb_to_hex(self, rgb):
        return "#" + "".join(f"{c:02x}" for c in rgb)

if __name__ == "__main__":
    root = tk.Tk()
    app = NPYImageEditor(root)
    root.mainloop()
