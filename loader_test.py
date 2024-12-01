from pathlib import Path

# Import the DataLoader class
from data_loader import DataLoader  # Replace 'your_module_name'

def main():
    # Set dataset path
    data_dir = Path("C:/Users/Lona/Desktop/Enhanced Artifact/data_dir")

    
    # Initialize DataLoader
    batch_size = 16
    data_loader = DataLoader(data_dir=data_dir, batch_size=batch_size)
    
    # Check if samples are loaded
    print(f"Number of samples: {len(data_loader.samples)}")
    if len(data_loader.samples) == 0:
        print("No samples found. Check your dataset path or words.txt file.")
        return
    
    # Print first few samples
    print("First few samples:")
    for sample in data_loader.samples[:5]:
        print(sample)
    
    # Test batch loading
    if data_loader.has_next():
        print("Loading a batch...")
        batch = data_loader.get_next()
        print(f"Batch size: {batch.batch_size}")
        print(f"First few ground truth texts: {batch.gt_texts[:5]}")

if __name__ == "__main__":
    main()