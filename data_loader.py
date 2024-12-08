import random
from collections import namedtuple
from typing import Tuple
import numpy as np
from pathlib import Path
import cv2

# Named tuples for samples and batches
Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size, img_paths')

class DataLoader:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self, data_dir: Path, batch_size: int, data_split: float = 0.90, fast: bool = True) -> None:
        """Initialize the data loader."""
        assert data_dir.exists(), "Data directory does not exist."

        self.fast = fast
        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []

        # Load dataset information
        f = open(data_dir / 'gt/words.txt')
        chars = set()
        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # Known broken images in IAM dataset
        for line in f:
            # Ignore empty and comment lines
            line = line.strip()
            if not line or line[0] == '#':
                continue

            # Split the line into parts
            line_split = line.split(' ')
            assert len(line_split) >= 9, f"Invalid line format: {line}"

            # Generate file path for the image
            file_name_split = line_split[0].split('-')
            file_name_subdir1 = file_name_split[0]
            file_name_subdir2 = f'{file_name_split[0]}-{file_name_split[1]}'
            file_base_name = line_split[0] + '.png'
            file_name = data_dir / 'imgs' / file_name_subdir1 / file_name_subdir2 / file_base_name

            if line_split[0] in bad_samples_reference:
                print('Ignoring known broken image:', file_name)
                continue

            # Extract ground truth text
            gt_text = ' '.join(line_split[8:])
            chars = chars.union(set(gt_text))

            # Add sample
            self.samples.append(Sample(gt_text, file_name))

        # Split into training and validation sets
        split_idx = int(data_split * len(self.samples))
        self.train_samples = self.samples[:split_idx]
        self.validation_samples = self.samples[split_idx:]

        print(f"Training samples: {len(self.train_samples)}")
        print(f"Validation samples: {len(self.validation_samples)}")

        # Collect character list and set initial training set
        self.char_list = sorted(list(chars))
        self.train_set()

    def train_set(self) -> None:
        """Switch to training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int, int]:
        """Return current batch index, total batches, and remaining batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # Full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # Allow smaller last batch

        curr_batch = self.curr_idx // self.batch_size + 1
        remaining_batches = num_batches - curr_batch + 1
        return curr_batch, num_batches, remaining_batches

    def has_next(self) -> bool:
        """Check if there are more batches to process."""
        return self.curr_idx < len(self.samples)

    def _get_img(self, i: int) -> np.ndarray:
        """Load image with OpenCV."""
        img_path = self.samples[i].file_path
        print(f"Loading image: {img_path}")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image file not found at {img_path}")
        return img

    def get_next(self) -> Batch:
        """Get the next batch of images, ground truth texts, and file paths."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]
        img_paths = [str(self.samples[i].file_path) for i in batch_range]  # Include image paths

        self.curr_idx += self.batch_size
        return Batch(imgs=imgs, gt_texts=gt_texts, batch_size=len(imgs), img_paths=img_paths)

