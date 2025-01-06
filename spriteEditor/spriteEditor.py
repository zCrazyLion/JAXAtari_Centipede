import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter.colorchooser import askcolor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.draw import rectangle

class NPYImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("NPY Image Editor")

        self.image = None
        self.zoom_level = 1
        self.selection_start = None
        self.selection_end = None
        self.current_color = [0, 0, 0]  # Default color: black
        
        self.create_widgets()

    def create_widgets(self):
        # Menu
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save Selection", command=self.save_selection)
        file_menu.add_command(label="Exit", command=self.root.quit)

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
        except Exception as e:
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

    def open_color_palette(self, event):
        color = askcolor(color=self.rgb_to_hex(self.current_color), title="Choose Color")
        if color[0]:
            self.current_color = [int(c) for c in color[0]]
            self.color_indicator.config(bg=self.rgb_to_hex(self.current_color))

    def on_mouse_press(self, event):
        if self.image is None or event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if self.tool == "pencil":
            self.image[y, x] = self.current_color
            self.update_display()
        elif self.tool == "rectangular_selection":
            self.selection_start = (y, x)
        elif self.tool == "dropper":
            self.current_color = self.image[y, x].tolist()
            self.color_indicator.config(bg=self.rgb_to_hex(self.current_color))

    def on_mouse_release(self, event):
        if self.tool == "rectangular_selection" and event.xdata and event.ydata:
            self.selection_end = (int(event.ydata), int(event.xdata))
            self.update_display()

    def on_mouse_motion(self, event):
        if self.tool == "pencil" and event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            self.image[y, x] = self.current_color
            self.update_display()

    def update_display(self):
        if self.image is None:
            return

        self.ax.clear()
        zoomed_image = self.image[::int(1/self.zoom_level), ::int(1/self.zoom_level)]
        self.ax.imshow(zoomed_image)

        if self.selection_start and self.selection_end:
            y0, x0 = self.selection_start
            y1, x1 = self.selection_end
            rect = rectangle(start=(min(y0, y1), min(x0, x1)), extent=(abs(y1 - y0), abs(x1 - x0)))
            self.ax.plot(rect[1], rect[0], color="red")

        self.canvas.draw()

    def rgb_to_hex(self, rgb):
        return "#" + "".join(f"{c:02x}" for c in rgb)

if __name__ == "__main__":
    root = tk.Tk()
    app = NPYImageEditor(root)
    root.mainloop()
