import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ---------------- CONFIG ---------------- #

MODEL_PATH = "models/tsynet_ortho_predictor.h5"

IMAGE_SIZE = (128, 128)

# Your actual folder names
class_names = ["normal", "overbite", "underbite"]

# Load trained model
model = load_model(MODEL_PATH, compile=False)

# ---------------- IMAGE PREPROCESSING ---------------- #

def prepare_image(image_path):
    
    # Load image in grayscale
    img = load_img(
        image_path,
        color_mode='grayscale',
        target_size=IMAGE_SIZE
    )

    # Convert image to array
    img_array = img_to_array(img)

    # Normalize image
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ---------------- MAIN APP ---------------- #

class OrthodonticPredictorApp:

    def __init__(self, root):

        self.root = root
        self.root.title("TSYNET Orthodontic Growth Predictor")
        self.root.geometry("700x750")
        self.root.configure(bg="#f0f4f7")

        self.image_path = None

        # -------- TITLE -------- #

        title = tk.Label(
            root,
            text="TSYNET Orthodontic Growth Predictor",
            font=("Arial", 20, "bold"),
            bg="#f0f4f7",
            fg="#003366"
        )

        title.pack(pady=15)

        # -------- LOAD BUTTON -------- #

        self.btn_load = tk.Button(
            root,
            text="Load X-ray Image",
            command=self.load_image,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=15,
            pady=8
        )

        self.btn_load.pack(pady=10)

        # -------- IMAGE DISPLAY -------- #

        self.canvas = tk.Canvas(
            root,
            width=450,
            height=450,
            bg="white",
            highlightthickness=2,
            highlightbackground="#cccccc"
        )

        self.canvas.pack(pady=10)

        # -------- PREDICT BUTTON -------- #

        self.btn_predict = tk.Button(
            root,
            text="Predict",
            command=self.predict,
            state=tk.DISABLED,
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=8
        )

        self.btn_predict.pack(pady=15)

        # -------- RESULT LABEL -------- #

        self.label_result = tk.Label(
            root,
            text="",
            font=("Arial", 16, "bold"),
            bg="#f0f4f7",
            fg="#222222",
            justify=tk.CENTER
        )

        self.label_result.pack(pady=15)

    # ---------------- LOAD IMAGE ---------------- #

    def load_image(self):

        filetypes = (
            ("Image files", "*.bmp *.jpg *.jpeg *.png"),
            ("All files", "*.*")
        )

        path = filedialog.askopenfilename(
            title="Select X-ray Image",
            filetypes=filetypes
        )

        if path:

            self.image_path = path

            self.show_image(path)

            self.label_result.config(text="")

            self.btn_predict.config(state=tk.NORMAL)

    # ---------------- SHOW IMAGE ---------------- #

    def show_image(self, path):

        img = Image.open(path)

        img.thumbnail((450, 450))

        self.photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")

        self.canvas.create_image(
            225,
            225,
            image=self.photo
        )

    # ---------------- PREDICT ---------------- #

    def predict(self):

        if not self.image_path:

            messagebox.showwarning(
                "Warning",
                "Please load an image first."
            )

            return

        try:

            # Prepare image
            img_array = prepare_image(self.image_path)

            # Predict
            preds = model.predict(img_array)[0]

            # Highest probability class
            class_index = np.argmax(preds)

            class_name = class_names[class_index]

            confidence = np.max(preds)

            # Reject low-confidence predictions
            if confidence < 0.30:

                self.label_result.config(
                    text="Invalid X-ray Image"
                )

                return

            # Show prediction
            self.label_result.config(
                text=f"Prediction: {class_name.upper()}\nConfidence: {confidence:.2f}"
            )

        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Prediction failed:\n{e}"
            )

# ---------------- RUN APP ---------------- #

if __name__ == "__main__":

    root = tk.Tk()

    app = OrthodonticPredictorApp(root)

    root.mainloop()