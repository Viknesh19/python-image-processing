import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
import urllib.request
from PIL import ImageTk, Image, ImageOps

# Global variables
original_image = None
processed_image = None
denoise_strength = 0.0
sharpen_strength = 0.0
display_width = 400
display_height = 400

def open_image():
    global original_image, processed_image

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        image = cv2.imread(file_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = original_image.copy()
        denoise_and_sharpen_images()
        display_images()

def open_image_from_url():
    global original_image, processed_image

    url = entry_url.get()
    if url:
        try:
            response = urllib.request.urlopen(url)
            image_data = response.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            processed_image = original_image.copy()
            denoise_and_sharpen_images()
            display_images()
        except Exception as e:
            print("Error:", str(e))

def denoise_image(value):
    global denoise_strength
    denoise_strength = round(float(value), 1)
    denoise_label.config(text=f"Denoise Strength: {denoise_strength}")
    denoise_and_sharpen_images()
    display_images()

def sharpen_image(value):
    global sharpen_strength
    sharpen_strength = round(float(value), 1)
    sharpen_label.config(text=f"Sharpen Strength: {sharpen_strength}")
    denoise_and_sharpen_images()
    display_images()

def denoise_and_sharpen_images():
    global original_image, processed_image, denoise_strength, sharpen_strength
    processed_image = apply_filters(original_image, denoise_strength)
    processed_image = apply_sharpening(processed_image, sharpen_strength)

def apply_filters(image, strength):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    return denoised_image

def apply_sharpening(image, strength):
    blurred_image = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened_image = cv2.addWeighted(image, 1 + strength, blurred_image, -strength, 0)
    return sharpened_image

def revert_image():
    global processed_image, denoise_strength, sharpen_strength
    if original_image is not None:
        processed_image = original_image.copy()
        denoise_slider.set(0)  # Reset denoise slider value
        sharpen_slider.set(0)  # Reset sharpen slider value
        display_images()

def display_images():
    global original_image, processed_image

    # Resize and pad original image to fit within the display size while maintaining aspect ratio
    original_pil = Image.fromarray(original_image)
    original_pil.thumbnail((display_width, display_height), Image.ANTIALIAS)
    original_pil = ImageOps.pad(original_pil, (display_width, display_height))

    # Resize processed image to fit within the display size while maintaining aspect ratio
    processed_pil = Image.fromarray(processed_image)
    processed_pil.thumbnail((display_width, display_height), Image.ANTIALIAS)
    processed_pil = ImageOps.pad(processed_pil, (display_width, display_height))
    
    # Convert images to Tkinter PhotoImage format
    original_image_tk = ImageTk.PhotoImage(original_pil)
    processed_image_tk = ImageTk.PhotoImage(processed_pil)

    # Update the image labels
    original_label.config(image=original_image_tk)
    processed_label.config(image=processed_image_tk)

    # Keep references to PhotoImage objects to prevent them from being garbage collected
    original_label.image = original_image_tk
    processed_label.image = processed_image_tk

def save_image():
    global processed_image
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
        if file_path:
            processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, processed_image_bgr)
            print("Image saved successfully.")




# Create GUI window
window = tk.Tk()
window.title("Image Processing")
window.geometry("900x600")
window.config(bg="black")

# Create top frame
top_frame = tk.Frame(window, bg='grey30')
top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2.5)

# Create left frame
left_frame = tk.Frame(window, bg='grey30')
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=2.5)

# Create right frame
right_frame = tk.Frame(window, bg='grey30')
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=2.5)


# Create "Open Image" button
open_button = tk.Button(top_frame, text="Open Image", command=open_image)
open_button.pack(fill=tk.BOTH, pady=5, padx=15)

# Create "Open Image from URL" button
url_frame = tk.Frame(top_frame, bg='grey')
url_frame.pack(fill=tk.BOTH)
url_button = tk.Button(url_frame, text="Open Image from URL", command=open_image_from_url)
url_button.pack(side=tk.RIGHT, fill=tk.BOTH, pady=5, padx=15)

# Create URL Entry field
entry_url = tk.Entry(url_frame)
entry_url.pack( fill=tk.BOTH, pady=10, padx=10)


# Create frames and labels in left_frame
tk.Label(left_frame, text="Original Image").pack(fill=tk.BOTH, padx=5, pady=5)

# Create ori image labels
original_label = tk.Label(left_frame)
original_label.pack(padx=10, pady=10)

# Create frames and labels in right_frame
tk.Label(right_frame, text="Edited Image").pack(fill=tk.BOTH, padx=5, pady=5)

# Create image labels
processed_label = tk.Label(right_frame)
processed_label.pack(padx=10, pady=10)

# Create tool bar frame
tool_bar = tk.Frame(left_frame)
tool_bar.pack(padx=10, pady=10)

# Create Denoise Slider
denoise_frame = tk.Frame(tool_bar)
denoise_frame.pack(pady=10, padx=10)

denoise_label = tk.Label(denoise_frame, text="Denoise Strength:")
denoise_label.pack(side=tk.LEFT)

denoise_slider = ttk.Scale(
    denoise_frame,
    from_=0,
    to=50,
    value=denoise_strength,
    orient=tk.HORIZONTAL,
    command=denoise_image
)
denoise_slider.pack(side=tk.LEFT, padx=10)

# Create Sharpen Slider
sharpen_frame = tk.Frame(tool_bar)
sharpen_frame.pack(pady=10, padx=10)

sharpen_label = tk.Label(sharpen_frame, text="Sharpen Strength:")
sharpen_label.pack(side=tk.LEFT)

sharpen_slider = ttk.Scale(
    sharpen_frame,
    from_=-3.0,
    to=3.0,
    value=sharpen_strength,
    orient=tk.HORIZONTAL,
    command=sharpen_image
)
sharpen_slider.pack(side=tk.LEFT, pady=10, padx=10)

# Create "Revert Image" button
revert_button = tk.Button(tool_bar, text="Revert Image", command=revert_image)
revert_button.pack(fill=tk.BOTH, pady=5, padx=10)

# Create "Save Image" button
save_button = tk.Button(tool_bar, text="Save Image", command=save_image)
save_button.pack(fill=tk.BOTH, pady=5, padx=10)

# Configure row and column weights for window and frames
window.rowconfigure(0, weight=0)
window.rowconfigure(1, weight=1)
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=3)
left_frame.columnconfigure(0, weight=1)
right_frame.columnconfigure(0, weight=1)


# Start the GUI event loop
window.mainloop()
