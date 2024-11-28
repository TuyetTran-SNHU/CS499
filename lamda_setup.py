'''
# not a good module to use to serializing/deserializing image
import pickle
#replace argparse by adding the file in the system directory
import argparse 
'''
import cv2
import lmdb
from path import Path

'''
# using import argparse to locate the path of a file
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()
'''

# Read the relative path to the images folder from a text file
with open('image_path.txt', 'r') as f:
    relative_path = f.read().strip()  # Read and remove any extra whitespace

# Get the absolute path relative to the current script's directory
data_dir = Path(__file__).parent / relative_path

# 5GB is enough for IAM dataset
assert not (data_dir / 'lmdb').exists()  # Updated to use data_dir instead of args.data_dir
env = lmdb.open(str(data_dir / 'lmdb'), map_size=1024 * 1024 * 1024 * 5)

# Go over all PNG files
# Updated to use data_dir instead of args.data_dir
fn_imgs = list((data_dir / 'img').walkfiles('*.png'))  

'''
# and put the imgs into lmdb as pickled grayscale imgs
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(i, len(fn_imgs))
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        basename = fn_img.basename()
        txn.put(basename.encode("ascii"), pickle.dumps(img))

env.close()
'''

# Start lambda enviroment as txn 
with env.begin(write=True) as txn:
    # loop through each image in the dir
    for i, fn_img in enumerate(fn_imgs):
        print(i, len(fn_imgs))
        
        # Read the image in grayscale
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        basename = fn_img.basename()
        
        # Encode the image as PNG
        success, buffer = cv2.imencode('.png', img)

        # Check if the encoding was successful before storing
        if success:
            txn.put(basename.encode("ascii"), buffer.tobytes())
        else:
            print("Encoding failed for image:", basename)

# Close the LMDB environment
env.close()
