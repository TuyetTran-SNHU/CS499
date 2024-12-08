from pathlib import Path
from data_loader import DataLoader  
from data_preprocessor import Preprocessor  
from model import Model 
import time


# Configuration
data_dir = Path("/Enhanced Artifact/data_dir")
batch_size = 16
img_size = (256, 64)  
epochs = 5  
char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- '\"!`~@#$%^&*()_=+:;[]1234567890?/.,><")
blank_token = '<blank>'
if blank_token not in char_list:
    char_list.append(blank_token)
print(f"Character list: {char_list}")
print(f"Blank token index: {len(char_list) - 1}")

def initialize_data_loader(data_dir, batch_size):
    """Initialize the data loader."""
    try:
        data_loader = DataLoader(data_dir=data_dir, batch_size=batch_size)
        if not data_loader.samples:
            raise ValueError("No samples found in the dataset. Check your dataset path or words.txt file.")
        
        print(f"Number of samples: {len(data_loader.samples)}")
        total_batches = (len(data_loader.samples) + batch_size - 1) // batch_size
        print(f"Number of batches: {total_batches}")
        
        print("First few samples:")
        for sample in data_loader.samples[:5]:
            print(sample)
        
        return data_loader
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        return None

def train_model(data_loader, model, preprocessor, epochs):
    """Train the model using the data loader and preprocess batches."""
    print("\nStarting training process...")
    try:
        model.train(data_loader, preprocessor, epochs)
        model.save("my_model.h5")
        print("Model saved to my_model.h5")
    except Exception as e:
        print(f"Error during training: {e}")

def test_model(data_loader, model, preprocessor):
    """Test the model on a validation set and display probabilities."""
    print("\nTesting the model...")
    try:
        data_loader.validation_set()  
        while data_loader.has_next():
            batch = data_loader.get_next()
            processed_batch = preprocessor.process_batch(batch)  
            predictions, probabilities = model.infer_batch(processed_batch.imgs, calc_probabilities=True)
            
            print("\nInference results:")
            for i, (gt, pred, prob, img_path) in enumerate(
                zip(processed_batch.gt_texts, predictions, probabilities, processed_batch.img_paths)
            ):
                print(f"Sample {i + 1}:")
                print(f"  Image Path: {img_path}")
                print(f"  Ground truth: {gt}")
                print(f"  Prediction: {pred}")
                print(f"  Probability: {prob:.2f}\n")
    except Exception as e:
        print(f"Error during testing: {e}")

def main():
    """Main function to run the training and testing pipeline."""
    start_time = time.time()

    # Initialize DataLoader
    data_loader = initialize_data_loader(data_dir, batch_size)
    if not data_loader:
        return

    # Initialize Preprocessor
    preprocessor = Preprocessor(img_size=img_size, data_augmentation=True)
    print(f"Preprocessor initialized with data_augmentation={preprocessor.data_augmentation}")

    # Initialize Model
    model = Model(char_list=char_list)
    model.compile_model()

    # Train the model
    train_model(data_loader, model, preprocessor, epochs)

    # Reload the model and test
    model.load("my_model.h5")
    print("Loaded my model for testing.")
    test_model(data_loader, model, preprocessor)

    # End timer and print total time
    elapsed_time = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
