from pathlib import Path
from data_loader import DataLoader  
from data_preprocessor import Preprocessor  
from model import Model 
import time
import cv2


# Configuration
data_dir = Path("./data_dir")
batch_size = 16
img_size = (256, 64)  
epochs = 1  
char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.-'\"!`~@#$%^&*()_=+:;[]1234567890?/.,><")
blank_token = '<blank>'
if blank_token not in char_list:
    char_list.append(blank_token)
print(f"Character list: {char_list}")
print(f"Blank token index: {len(char_list) - 1}")

def initialize_data_loader(data_dir, batch_size):
    
    """Initialize the data loader.
    Total valid 'ok' samples: 4189
    Training samples: 3770
    Validation samples: 419
    Training samples: 3770
    Training batches: 236
    Validation samples: 419
    Validation batches: 27
    DataLoader initialized successfully."""
    try:
        data_loader = DataLoader(data_dir=data_dir, batch_size=batch_size, img_size=img_size)

        # Training set
        data_loader.train_set()
        print(f"Training samples: {len(data_loader.samples)}")
        total_batches = (len(data_loader.samples) + batch_size - 1) // batch_size
        print(f"Training batches: {total_batches}")

        # Validation set
        data_loader.validation_set()
        print(f"Validation samples: {len(data_loader.samples)}")
        total_batches = (len(data_loader.samples) + batch_size - 1) // batch_size
        print(f"Validation batches: {total_batches}")

        return data_loader
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        return None

def train_model(data_loader, model, preprocessor, epochs):
    """Train the model using the data loader and preprocess batches."""
    print("\nStarting training process...")
    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_samples = 0
            correct_predictions = 0
            total_loss = 0
            
            data_loader.train_set()  # Switch to training data
            while data_loader.has_next():
                batch = data_loader.get_next()
                processed_batch = preprocessor.process_batch(batch)

                # Train on the batch
                loss, batch_correct = model.train_batch(processed_batch.imgs, processed_batch.gt_texts)
                
                # Update total metrics
                total_loss += loss
                total_samples += len(processed_batch.gt_texts)
                correct_predictions += batch_correct

                # Log intermediate results
                print(f"Batch loss: {loss:.4f}, Batch accuracy: {(batch_correct / len(processed_batch.gt_texts)) * 100:.2f}%")
            
            # Calculate epoch metrics
            avg_loss = total_loss / total_samples
            epoch_accuracy = (correct_predictions / total_samples) * 100
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            print(f"Epoch {epoch + 1} accuracy: {epoch_accuracy:.2f}%")
        
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

    # Initialize LMDB Path
    lmdb_path = data_dir / "lmdb"
    print(f"Using LMDB path: {lmdb_path}")

    # Check if LMDB exists, create if necessary
    if not lmdb_path.exists():
        print("LMDB database not found. Creating a new database...")
        preprocessor = Preprocessor(img_size=img_size, data_augmentation=False, lmdb_path=lmdb_path)
        data_loader = initialize_data_loader(data_dir, batch_size)
        if not data_loader:
            return
        # Populate LMDB with preprocessed images
        for sample in data_loader.samples:
            img = cv2.imread(str(sample.file_path), cv2.IMREAD_GRAYSCALE)
            preprocessor.add_to_lmdb(sample.file_path.name, img)
        preprocessor.close_lmdb()
        print("LMDB database created successfully.")
    else:
        print("LMDB database found. Using the existing database.")

    # Initialize Preprocessor with LMDB
    preprocessor = Preprocessor(img_size=img_size, data_augmentation=True)
    print(f"Preprocessor initialized with data_augmentation={preprocessor.data_augmentation}")

    # Initialize DataLoader with LMDB
    data_loader = DataLoader(data_dir=data_dir, batch_size=batch_size, img_size=img_size)
    if not data_loader:
        return

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