import random
from collections import namedtuple
from typing import List, Tuple
import numpy as np
from pathlib import Path
import cv2
import lmdb

# Named tuples for samples and batches
Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size, img_paths')

class DataLoader:
    def __init__(self, data_dir: Path, batch_size: int, img_size: Tuple[int, int], data_split: float = 0.90, fast: bool = True) -> None:
        """Initialize the data loader with LMDB and file system integration."""
        assert data_dir.exists()

        self.fast = fast
        self.img_size = img_size
        self.curr_idx = 0
        self.batch_size = batch_size

        # Load samples from words.txt
        words_file_path = data_dir / 'gt/words.txt'
        self.samples = self._load_samples(words_file_path, data_dir)  # Ensure self.samples is populated

        # Split samples into train and validation sets
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        # Debug: Print number of samples in each split
        print(f"Training samples: {len(self.train_samples)}")
        print(f"Validation samples: {len(self.validation_samples)}")

        # Debug: Print top 5 samples from training and validation sets
        print("\nTop 5 Training Samples:")
        for i, sample in enumerate(self.train_samples[:5]):
            print(f"Sample {i + 1}: GT Text: {sample.gt_text}, File Path: {sample.file_path}")

        print("\nTop 5 Validation Samples:")
        for i, sample in enumerate(self.validation_samples[:5]):
            print(f"Sample {i + 1}: GT Text: {sample.gt_text}, File Path: {sample.file_path}")

        # Initialize LMDB if fast mode is enabled
        if fast:
            self.env = lmdb.open(str(data_dir / 'lmdb'), readonly=True)

            # DEBUG: Count and print the total number of keys in LMDB
            with self.env.begin() as txn:
                cursor = txn.cursor()
                total_keys = sum(1 for _ in cursor)  # Count all keys
            print(f"Total number of datasets (keys) in LMDB: {total_keys}")

            # DEBUG: Print only the first 5 keys stored in LMDB
            print("\nFirst 5 keys stored in LMDB:")
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for idx, (key, _) in enumerate(cursor):
                    if idx >= 5:  # Limit to the first 5 keys
                        break
                    print(f"  {key.decode('utf-8')}")  # Decode key to string for readability
            print("Finished listing LMDB keys.\n")

        # Start with training set
        self.train_set()
    def _load_samples(self, words_file_path: Path, data_dir: Path) -> List[Sample]:
        """
        Total valid 'ok' samples: 4189
        Training samples: 3770
        Validation samples: 419
        Load samples from words.txt, ensuring only 'ok' files are included.

        Args:
            words_file_path (Path): Path to the words.txt file.
            data_dir (Path): Root directory of the dataset.

        Returns:
            List[Sample]: List of valid samples with GT text and file paths.
        """
        samples = []
        missing_files = []

        with open(words_file_path, 'r') as f:
            for line in f:
                # Ignore comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split(' ')
                if 'ok' not in parts[1]:  # Only include lines marked as 'ok'
                    continue

                folder_structure = parts[0][:3]  # e.g., "a01"
                base_name = parts[0]  # e.g., "a01-003-00-00"

                # Determine the subfolder prefix
                if len(parts[0]) > 8 and parts[0][8].isalpha():
                    subfolder_prefix = parts[0][:9]  # Include the extra character
                else:
                    subfolder_prefix = parts[0][:8]  # Use the first 8 characters only

                # Remove trailing dash, if any
                subfolder_prefix = subfolder_prefix.rstrip('-')

                # Construct the subfolder path
                subfolder_path = data_dir / 'imgs' / folder_structure / subfolder_prefix

                # Check if the file exists in the constructed subfolder
                file_path = subfolder_path / f"{base_name}.png"
                if not file_path.exists():
                    missing_files.append(str(file_path))
                    continue

                # Construct the full file path and append the sample
                gt_text = ' '.join(parts[8:])
                samples.append(Sample(gt_text, file_path))

        # Log all missing files
        if missing_files:
            missing_log_path = data_dir / 'missing_files.log'
            with open(missing_log_path, 'w') as log_file:
                log_file.write("\n".join(missing_files))
            print(f"Missing files logged to: {missing_log_path}")

        # Debug: Print the total number of valid samples
        print(f"Total valid 'ok' samples: {len(samples)}")

        return samples

    def _get_img(self, file_path: Path) -> np.ndarray:
        """
        Retrieve an image from LMDB or the file system.

        Args:
            file_path (Path): Path to the image file.

        Returns:
            np.ndarray: Grayscale image.
        """
        if self.fast:
            # Use only the base name of the file as the key
            key = file_path.name  # Get file name, e.g., "a01-132x-08-02.png"
            with self.env.begin() as txn:
                buffer = txn.get(key.encode("utf-8"))
                if buffer is None:
                    print(f"Image not found in LMDB for key: {key}")
                    raise RuntimeError(f"Image not found in LMDB: {file_path}")
                img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            if not file_path.exists():
                raise RuntimeError(f"Image file not found: {file_path}")
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)
        return img

    def train_set(self) -> None:
        """Switch to training set."""
        self.samples = self.train_samples
        self.curr_idx = 0
        random.shuffle(self.samples)

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.samples = self.validation_samples
        self.curr_idx = 0

    def has_next(self) -> bool:
        """Check if there are more batches to process."""
        return self.curr_idx < len(self.samples)

    def get_next(self) -> Batch:
        """Get the next batch of samples."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))
        imgs = [self._get_img(self.samples[i].file_path) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        img_paths = [self.samples[i].file_path for i in batch_range]
        self.curr_idx += self.batch_size
        return Batch(imgs=imgs, gt_texts=gt_texts, batch_size=len(imgs), img_paths=img_paths)

# Example usage
if __name__ == "__main__":
    data_dir = Path("data_dir")  # Replace with the actual dataset directory
    batch_size = 8
    img_size = (64, 256)

    print("Initializing DataLoader...")
    data_loader = DataLoader(data_dir=data_dir, batch_size=batch_size, img_size=img_size)

    print("\nSwitching to training set...")
    data_loader.train_set()

    if data_loader.has_next():
        batch = data_loader.get_next()
        print(f"\nFetched a batch of {batch.batch_size} samples:")
        for i, (img, gt_text) in enumerate(zip(batch.imgs, batch.gt_texts)):
            print(f"Sample {i + 1}:")
            print(f"  Image shape: {img.shape}")
            print(f"  Ground truth text: {gt_text}")

    print("\nSwitching to validation set...")
    data_loader.validation_set()
