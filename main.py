from pathlib import Path
from data_loader import DataLoader, Batch  
from data_preprocessor import Preprocessor  # Import the Preprocessor class
from model import Model  

# Configuration
DATA_DIR = Path("C:/Users/Lona/Desktop/Enhanced Artifact/data_dir")  # Dataset path
BATCH_SIZE = 150
IMG_SIZE = (1000, 1000)  # Target image size (width, height)
EPOCHS = 1
CHAR_LIST = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.- '\"!`~@#$%^&*()_=+:;[]1234567890?/.,><")  # Recognizable characters

def initialize_data_loader(data_dir, batch_size):
    """Initialize the data loader."""
    data_loader = DataLoader(data_dir=data_dir, batch_size=batch_size)
    print(f"Number of samples: {len(data_loader.samples)}")
    if len(data_loader.samples) == 0:
        print("No samples found. Check your dataset path or words.txt file.")
        return None

    total_batches = (len(data_loader.samples) + batch_size - 1) // batch_size  # Ceiling division
    print(f"Number of batches: {total_batches}")

    print("First few samples:")
    for sample in data_loader.samples[:5]:
        print(sample)

    return data_loader

def train_model(data_loader, model, preprocessor, epochs):
    """Train the model using the data loader and preprocess batches."""
    print("\nStarting training process...")
    data_loader.train_set()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        while data_loader.has_next():
            batch = data_loader.get_next()
            processed_batch = preprocessor.process_batch(batch)  # Preprocess the batch
            model.train_batch(processed_batch.imgs, processed_batch.gt_texts, epochs=1)
    print("\nTraining completed successfully!")

def test_model(data_loader, model, preprocessor):
    """Test the model on a validation set."""
    print("\nTesting the model...")
    data_loader.validation_set()  # Switch to validation/test set

    if data_loader.has_next():
        batch = data_loader.get_next()
        processed_batch = preprocessor.process_batch(batch)  # Preprocess the batch
        predictions = model.infer_batch(processed_batch.imgs)

        print("\nInference results:")
        for i, (gt, pred) in enumerate(zip(processed_batch.gt_texts, predictions)):
            print(f"Ground truth: {gt}")
            print(f"Prediction: {pred}\n")

def main():
    """Main function to run the training and testing pipeline."""
    # Initialize DataLoader
    data_loader = initialize_data_loader(DATA_DIR, BATCH_SIZE)
    if not data_loader:
        return

    # Initialize Preprocessor
    preprocessor = Preprocessor(img_size=IMG_SIZE, data_augmentation=True)

    # Initialize Model
    model = Model(char_list=CHAR_LIST)
    model.compile_model()

    # Train the model
    train_model(data_loader, model, preprocessor, EPOCHS)

    # Test the model
    test_model(data_loader, model, preprocessor)

    # Save the trained model
    model.save("htr_model_snapshot")
    print("\nModel saved successfully.")

if __name__ == "__main__":
    main()
