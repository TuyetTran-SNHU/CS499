import numpy as np
import tensorflow as tf
from model import Model, char_list  # Import Model and char_list from model.py

def test_model():
    """Test the Model initialization and basic functionality."""
    print("Testing the Model from the imported script...")
    
    # Example character list and decoder type for testing
    char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    decoder_type = model.DecoderType.BestPath
    must_restore = False  # Set this based on whether you want to restore a model

    # Initialize the Model
    try:
        print("Initializing model for testing...")
        model = model.Model(char_list=char_list, decoder_type=decoder_type, must_restore=must_restore)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return

    # Setup TensorFlow session
    try:
        print("Setting up TensorFlow session...")
        sess, saver = model.setup_tf()
        print("TensorFlow session set up successfully.")
    except Exception as e:
        print(f"Error during TensorFlow session setup: {e}")
        return

    print("Test complete. Model is ready for further use.")

if __name__ == "__main__":
    test_model()