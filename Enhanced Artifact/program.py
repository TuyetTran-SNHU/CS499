import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os
import cv2
from model import Model
from data_preprocessor import Preprocessor

# Define the character list
char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- '\"!`~@#$%^&*()_=+:;[]1234567890?/.,><")
blank_token = '<blank>'
if blank_token not in char_list:
    char_list.append(blank_token)

# Initialize the model and load weights
htr_model = Model(char_list=char_list)
htr_model.load("my_model.h5")

# Initialize the preprocessor
preprocessor = Preprocessor(img_size=(256, 32), data_augmentation=False)

# Recognize handwriting
def recognize_handwriting(image_path):
    """Run the image through the model and return the prediction."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        preprocessed_img = preprocessor.process_img(img)
        preprocessed_img = preprocessed_img[np.newaxis, ...]  # Add batch dimension

        prediction = htr_model.model.predict(preprocessed_img)
        decoded_prediction = htr_model.decode_predictions(prediction)
        return decoded_prediction

    except Exception as e:
        print(f"Error processing image: {e}")
        raise

# GUI setup
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png")])
    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img.thumbnail((400, 200))
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)
        image_label.image = img_display

        result = recognize_handwriting(file_path)
        result_label.config(text=f"Prediction: {result}")

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        result_label.config(text=f"Error: {str(e)}")

root = tk.Tk()
root.title("Handwriting Recognition")
root.geometry("600x400")

frame = tk.Frame(root, width=600, height=300)
frame.pack_propagate(False)
frame.pack()

image_label = tk.Label(frame)
image_label.pack()

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

result_label = tk.Label(root, text="Prediction will appear here.", font=("Arial", 14))
result_label.pack()

root.mainloop()
