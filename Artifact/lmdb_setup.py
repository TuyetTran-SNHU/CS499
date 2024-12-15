import argparse
import cv2
import lmdb
from path import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()

# 2GB is enough for IAM dataset
assert not (args.data_dir / 'lmdb').exists()
env = lmdb.open(str(args.data_dir / 'lmdb'), map_size=1024 * 1024 * 1024 * 2)

# Go over all PNG files
fn_imgs = list((args.data_dir / 'imgs').walkfiles('*.png'))

# Store the images into LMDB as encoded PNG data
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(f"Processing image {i + 1}/{len(fn_imgs)}: {fn_img}")

        # Read the image in grayscale
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {fn_img}, skipping.")
            continue

        # Encode the image to PNG format
        success, buffer = cv2.imencode('.png', img)
        if not success:
            print(f"Error: Encoding failed for image {fn_img.name}, skipping.")
            continue

        # Store the encoded image in LMDB
        key = fn_img.basename()  # Use the base name as the key
        txn.put(key.encode("ascii"), buffer.tobytes())

print("LMDB creation completed successfully.")
env.close()


'''def setup_lmdb(data_dir: Path, lmdb_dir: Path) -> None:
    """Create LMDB from images."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Error: Data directory does not exist at {data_dir}")
    if lmdb_dir.exists():
        raise FileExistsError(f"Error: LMDB directory already exists at {lmdb_dir}")

    # Open LMDB environment
    env = lmdb.open(str(lmdb_dir), map_size=1024 * 1024 * 1024 * 2)  # 2GB map size
    fn_imgs = list(data_dir.rglob('*.png'))  # Find all PNG files recursively

    if not fn_imgs:
        print("Warning: No PNG files found in the 'imgs' folder.")

    # Debug: Print the number of found images
    print(f"Found {len(fn_imgs)} PNG files in {data_dir}")

    # Start LMDB transaction
    with env.begin(write=True) as txn:
        for i, fn_img in enumerate(fn_imgs):
            print(f"Processing image {i + 1}/{len(fn_imgs)}: {fn_img}")

            # Read the image in grayscale
            img = cv2.imread(str(fn_img), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image {fn_img}, skipping.")
                continue

            # Encode the image
            success, buffer = cv2.imencode('.png', img)
            if not success:
                print(f"Error: Encoding failed for image {fn_img.name}, skipping.")
                continue

            # Store the encoded image in LMDB
            key = fn_img.name  # Use the base name as the key
            txn.put(key.encode("ascii"), buffer.tobytes())

    print("LMDB creation completed successfully.")

def verify_lmdb(lmdb_dir: Path, keys_to_check: list) -> None:
    """Verify images stored in LMDB."""
    if not lmdb_dir.exists():
        raise FileNotFoundError(f"Error: LMDB directory does not exist at {lmdb_dir}")

    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False)

    print(f"Verifying images in LMDB at {lmdb_dir}")

    with env.begin(write=False) as txn:
        for key in keys_to_check:
            print(f"Retrieving key: {key}")
            buffer = txn.get(key.encode("ascii"))
            if buffer is None:
                print(f"Error: Key {key} not found in LMDB")
                continue

            # Decode the image
            img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Decoding failed for key {key}")
            else:
                print(f"Image retrieved successfully for key {key}, shape: {img.shape}")
                # Optional: Display the image for debugging
                # cv2.imshow(f"Image: {key}", img)
                # cv2.waitKey(0)

    # Optional: Close all OpenCV windows
    # cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Set up paths
    data_dir = Path("data_dir/imgs")  # Path to the images folder
    lmdb_dir = Path("data_dir/imgs/lmdb")  # Path to the LMDB folder

    # Step 1: Create LMDB
    setup_lmdb(data_dir=data_dir, lmdb_dir=lmdb_dir)

    # Step 2: Verify LMDB
    keys_to_check = ["a01-000u-06-02.png", "a01-000u-06-03.png"]  # Replace with actual keys
    verify_lmdb(lmdb_dir=lmdb_dir, keys_to_check=keys_to_check)'''


'''# Explicitly define the relative path to the images folder
relative_path = "data_dir/imgs"

# Resolve the absolute path relative to the current script's directory
data_dir = Path(__file__).parent / relative_path

# Debug: Print resolved data directory
print("Resolved path to data directory:", data_dir)

# Ensure the data directory exists
if not data_dir.exists():
    raise FileNotFoundError(f"Error: Data directory does not exist at {data_dir}")

# Path to the words.txt file
words_file_path = data_dir.parent / "gt" / "words.txt"
if not words_file_path.exists():
    raise FileNotFoundError(f"Error: 'words.txt' does not exist at {words_file_path}")

# Debug: Read and print the top 5 valid lines from words.txt
valid_lines = []
with open(words_file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):  # Skip empty lines and comments
            valid_lines.append(line)
            if len(valid_lines) == 5:
                break

print("Top 5 valid lines from words.txt:")
for line in valid_lines:
    print(line)

# Open LMDB environment
lmdb_dir = data_dir / "lmdb"
if lmdb_dir.exists():
    raise FileExistsError(f"Error: LMDB directory already exists at {lmdb_dir}")
env = lmdb.open(str(lmdb_dir), map_size=1024 * 1024 * 1024 * 2)  # 2GB map size

# Find all PNG files in the 'imgs' directory
fn_imgs = list(data_dir.rglob('*.png'))  # rglob is used for recursive search

if not fn_imgs:
    print("Warning: No PNG files found in the 'imgs' folder.")

# Debug: Print the number of found images
print(f"Found {len(fn_imgs)} PNG files in {data_dir}")

# Start LMDB transaction
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(f"Processing image {i + 1}/{len(fn_imgs)}: {fn_img}")

        # Read the image in grayscale
        img = cv2.imread(str(fn_img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {fn_img}, skipping.")
            break

        # Test encoding and decoding
        success, buffer = cv2.imencode('.png', img)
        if not success:
            print(f"Error: Encoding failed for image {fn_img.name}, break.")
            break

        decoded_img = cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded_img is None:
            print(f"Error: Decoding failed for image {fn_img.name}, break.")
            break

        print(f"Image processed successfully for {fn_img.name}, shape: {decoded_img.shape}")

        # Store the encoded image in LMDB
        key = fn_img.name  # Use the base name as the key
        txn.put(key.encode("ascii"), buffer.tobytes())

print("LMDB creation completed successfully.")'''