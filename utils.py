import tensorflow as tf
import numpy as np

def resize_with_padding(img, target_height, target_width):
    """Resizes image to target size while maintaining aspect ratio and adding padding."""
    # Ensure the image has 3 dimensions (height, width, channels)
    if len(img.shape) == 2:  # This checks if the image is 2D (height, width)
        img = tf.expand_dims(img, axis=-1)  # Add a channel dimension (height, width, 1)

    # Resize image while preserving aspect ratio (nearest neighbor interpolation)
    img = tf.image.resize(img, (target_height, target_width), method='nearest')

    # Padding to make it exactly the target size
    padded_img = tf.image.resize_with_crop_or_pad(img, target_height, target_width)
    return padded_img

def texts_to_sparse(texts, char_list):
    """Convert text labels to a sparse tensor for CTC."""
    indices, values, max_len = [], [], 0
    for batch_idx, text in enumerate(texts):
        label = [char_list.index(char) for char in text]
        max_len = max(max_len, len(label))
        for pos, val in enumerate(label):
            indices.append([batch_idx, pos])
            values.append(val)
    dense_shape = [len(texts), max_len]
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
