import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

# Initialize YOLO model
model = YOLO("best.pt")

def upload_image():
    """Function to upload an image file."""
    global img_path, before_img_label

    img_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if img_path:
        # Load and display the image in the "Before Detection" section
        img = Image.open(img_path)
        img.thumbnail((400, 400))  # Resize image for display
        img_tk = ImageTk.PhotoImage(img)
        before_img_label.config(image=img_tk)
        before_img_label.image = img_tk
        messagebox.showinfo("Image Upload", "Image uploaded successfully!")

def run_yolo():
    """Function to run YOLO prediction on the uploaded image."""
    if not img_path:
        messagebox.showerror("Error", "Please upload an image first!")
        return

    # Run YOLO prediction (disable saving results to disk)
    results = model.predict(source=img_path, save=False, conf=0.5)

    # Extract the first result (image with annotations)
    result_img_array = results[0].plot()  # Get annotated image as a numpy array

    # Convert numpy array to PIL Image for Tkinter compatibility
    result_img = Image.fromarray(cv2.cvtColor(result_img_array, cv2.COLOR_BGR2RGB))
    result_img.thumbnail((400, 400))  # Resize for display
    result_img_tk = ImageTk.PhotoImage(result_img)

    # Display the result in the "After Detection" section
    after_img_label.config(image=result_img_tk)
    after_img_label.image = result_img_tk
    messagebox.showinfo("YOLO Prediction", "Prediction completed successfully!")

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("YOLO Object Detection")
root.geometry("900x600")
root.config(bg="#f0f0f0")

# Header
header = tk.Label(root, text="YOLO Object Detection", bg="#0078D7", fg="white", font=("Arial", 24, "bold"), pady=10)
header.pack(fill="x")

# Main Frame (Centering the layout)
main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(expand=True)

# Before and After Frames
before_frame = tk.LabelFrame(main_frame, text="Before Detection", font=("Arial", 14, "bold"), bg="#f9f9f9", padx=10, pady=10, width=400, height=400)
before_frame.grid(row=0, column=0, padx=20, pady=10)
before_frame.grid_propagate(False)

after_frame = tk.LabelFrame(main_frame, text="After Detection", font=("Arial", 14, "bold"), bg="#f9f9f9", padx=10, pady=10, width=400, height=400)
after_frame.grid(row=0, column=1, padx=20, pady=10)
after_frame.grid_propagate(False)

# Image Labels
before_img_label = tk.Label(before_frame, bg="#f9f9f9")
before_img_label.pack(expand=True)

after_img_label = tk.Label(after_frame, bg="#f9f9f9")
after_img_label.pack(expand=True)

# Controls Frame
controls_frame = tk.Frame(root, bg="#f0f0f0")
controls_frame.pack(fill="x", pady=10)

# Buttons
upload_button = tk.Button(controls_frame, text="Upload Image", command=upload_image, bg="#0078D7", fg="white", font=("Arial", 14), padx=10)
upload_button.pack(side="left", padx=20)

run_button = tk.Button(controls_frame, text="Run YOLO Detection", command=run_yolo, bg="#28a745", fg="white", font=("Arial", 14), padx=10)
run_button.pack(side="left", padx=20)

exit_button = tk.Button(controls_frame, text="Exit", command=root.quit, bg="#dc3545", fg="white", font=("Arial", 14), padx=10)
exit_button.pack(side="right", padx=20)

# Footer
footer = tk.Label(root, text="Developed by Chiaw Na, Syahirah, Amir , Shafa", bg="#0078D7", fg="white", font=("Arial", 12), pady=10)
footer.pack(fill="x")

# Start the Tkinter event loop
root.mainloop()
