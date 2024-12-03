import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path
from model import Model  # Assuming your model is saved in a file named model.py
from data_loader import DataLoader  # Assuming your data loader is saved in a file named data_loader.py

# Configuration
DATA_DIR = Path("./data_dir")  # Path to your IAM dataset
BATCH_SIZE = 150
IMG_SIZE = (256, 32)  # Target image size (height, width)
EPOCHS = 5
CHAR_LIST = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- '\"!`~@#$%^&*()_=+:;[]1234567890?/.,><")  # Characters the model can recognize

# Initialize the DataLoader for training
data_loader = DataLoader(DATA_DIR, batch_size=BATCH_SIZE)

# Initialize the Model
model = Model(char_list=CHAR_LIST)

# Compile the model
model.compile_model()

# Train the model
def train_model():
    # Start training using the training set
    data_loader.train_set()

    # Loop over epochs
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        # Iterate through the training data
        while data_loader.has_next():
            # Get the next batch of images and labels
            batch = data_loader.get_next()

            # Convert the batch labels into a suitable format (e.g., indices for characters)
            model.train_batch(batch.imgs, batch.gt_texts, epochs=1)

train_model()

# Inference: Test the model on a single batch (after training)
def test_model():
    # Assuming test set is available
    data_loader.validation_set()  # Switch to validation/test set

    # Get the first batch from the validation set
    batch = data_loader.get_next()

    # Run inference using the trained model
    predictions = model.infer_batch(batch.imgs)

    # Print out the results
    for i, (gt, pred) in enumerate(zip(batch.gt_texts, predictions)):
        print(f"Ground truth: {gt}")
        print(f"Prediction: {pred}\n")

# Run inference after training
test_model()

# Save the trained model
model.save("htr_model_snapshot")
