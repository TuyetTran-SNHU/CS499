import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

IMG_SIZE = (1000, 1000)  # Example (height, width)

class Model:
    """TensorFlow 2.x compatible model for Handwritten Text Recognition (HTR)."""

    def __init__(self, char_list):
        """Initialize the HTR model."""
        self.char_list = char_list
        print("Initializing model...")
        # Build the model
        self.model = self.build_model()
        self.model.summary()
        print("Model initialized successfully.")

    def build_model(self):
        """Build the CNN, RNN, and CTC architecture."""
        print("Building model...")
        # Input layer for images
        input_imgs = layers.Input(shape=(None, None, 1), name="input_imgs")  # (Batch, Height, Width, Channels)
        
        # CNN layers
        x = self.setup_cnn(input_imgs)

        # RNN layers
        x = self.setup_rnn(x)

        # Output layer for softmax predictions
        output = self.setup_ctc(x)

        print("Model built successfully.")
        return models.Model(inputs=input_imgs, outputs=output)

    def setup_cnn(self, input_tensor):
        """Build the CNN layers."""
        print("Setting up CNN layers...")
        x = input_tensor
        filters = [32, 64, 128, 128, 256]
        kernel_sizes = [5, 5, 3, 3, 3]
        pool_sizes = [(2, 2), (2, 2), (2, 2), (2, 2), (1, 2)]

        for filter_size, kernel_size, pool_size in zip(filters, kernel_sizes, pool_sizes):
            x = layers.Conv2D(filters=filter_size, kernel_size=kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D(pool_size=pool_size)(x)
        print("CNN setup complete.")
        return x

    def setup_rnn(self, cnn_output):
        """Build the RNN layers."""
        print("Setting up RNN layers...")
        x = layers.Reshape((-1, cnn_output.shape[-1]))(cnn_output)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        print("RNN setup complete.")
        return x

    def setup_ctc(self, rnn_output):
        """Add a dense layer for CTC."""
        print("Setting up CTC layer...")
        return layers.Dense(len(self.char_list) + 1, activation="softmax", name="ctc_output")(rnn_output)

    def compile_model(self):
        """Compile the model with CTC loss."""
        print("Compiling model...")
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=self.ctc_loss)
        print("Model compiled successfully.")

    def ctc_loss(self, y_true, y_pred):
        """Compute CTC loss."""
        
        # Compute input lengths (number of time steps for each sample in the batch)
        input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])  # [batch_size] (same time steps for all samples)
        
        # Compute label lengths (number of valid characters for each sample in the batch)
        label_length = tf.reduce_sum(tf.cast(y_true != -1, tf.int32), axis=1)  # [batch_size] (count non-padded labels)

        # Ensure that y_true is of the correct type (int32)
        y_true = tf.cast(y_true, tf.int32)  # CTC expects integer labels (indices)

        # Use tf.nn.ctc_loss to calculate the CTC loss
        loss = tf.reduce_mean(tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,  # set to False if you're using the shape [batch_size, time_steps, num_classes]
            blank_index=len(self.char_list)  # Assuming the blank index is after the last character in char_list
        ))
        
        return loss

    def train_batch(self, batch_imgs, batch_labels, epochs=1):
        """Train the model with a batch of images and labels."""
        print(f"Starting training for {epochs} epochs...")

        # Resize and pad images to a fixed target size (e.g., 256x32)
        target_height, target_width = IMG_SIZE  # e.g., (32, 256)

        resized_and_padded_imgs = [self.resize_with_padding(img, target_height, target_width) for img in batch_imgs]

        # Convert the list of processed images into a NumPy array
        batch_imgs = np.array(resized_and_padded_imgs)

        print(f"batch_imgs shape: {batch_imgs.shape}")  # Should be [batch_size, height, width, channels]

        # Validate shapes of inputs
        print("Validating shapes...")
        print("batch_imgs shape:", batch_imgs.shape)  # Should be [batch_size, height, width, channels]
        print("batch_labels:", batch_labels)         # Should be a list of strings
        for i, label in enumerate(batch_labels):
            print(f"Label {i}: '{label}' (length: {len(label)})")

        # Convert labels to sparse tensor
        sparse_labels = self.texts_to_sparse(batch_labels)

        # Ensure sparse_labels is created successfully
        if sparse_labels is not None:
            # Convert sparse tensor to dense representation with padding (-1)
            dense_labels = tf.sparse.to_dense(sparse_labels, default_value=-1)
            # Further validation
            print("sparse_labels shape:", sparse_labels.dense_shape.numpy())  # Sparse tensor shape
            print("dense_labels shape:", dense_labels.shape)  # Dense tensor shape
            
            # Ensure the batch size is consistent
            target_batch_size = batch_imgs.shape[0]
            if len(batch_imgs) < target_batch_size:
                batch_imgs, dense_labels = self.pad_last_batch(batch_imgs, dense_labels, target_batch_size)

            # Train the model (without drop_remainder since we're handling the last batch manually)
            self.model.fit(x=batch_imgs, y=dense_labels, epochs=epochs, batch_size=batch_imgs.shape[0])
            print("Training complete.")
        else:
            print("Error: Failed to convert labels to sparse tensor.")

    def pad_last_batch(self, batch_imgs, batch_labels, target_batch_size):
        """Pad the last batch if it's smaller than the batch size."""
        if len(batch_imgs) < target_batch_size:
            pad_size = target_batch_size - len(batch_imgs)
            
            # Pad images with zeros or an empty placeholder (for example)
            batch_imgs = np.pad(batch_imgs, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
            
            # Pad labels with -1 (or a suitable placeholder)
            batch_labels = np.pad(batch_labels, ((0, pad_size), (0, 0)), mode='constant', constant_values=-1)

        return batch_imgs, batch_labels


    def texts_to_sparse(self, texts):
        """Convert text labels to a sparse tensor for CTC."""
        indices, values, max_len = [], [], 0
        for batch_idx, text in enumerate(texts):
            label = [self.char_list.index(char) for char in text]
            max_len = max(max_len, len(label))
            for pos, val in enumerate(label):
                indices.append([batch_idx, pos])
                values.append(val)
        dense_shape = [len(texts), max_len]
        return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    def resize_with_padding(self, img, target_height, target_width):
        """Resizes image to target size while maintaining aspect ratio and adding padding."""
        # Ensure the image has 3 dimensions (height, width, channels)
        if len(img.shape) == 2:  # This checks if the image is 2D (height, width)
            img = tf.expand_dims(img, axis=-1)  # Add a channel dimension (height, width, 1)

        # Resize image while preserving aspect ratio (nearest neighbor interpolation)
        img = tf.image.resize(img, (target_height, target_width), method='nearest')

        # Padding to make it exactly the target size
        padded_img = tf.image.resize_with_crop_or_pad(img, target_height, target_width)
        return padded_img

    def infer_batch(self, batch_imgs):
        """Run inference on a batch of images."""
        print(f"batch_imgs shape: {batch_imgs.shape}")

        predictions = self.model.predict(batch_imgs)
        print(f"Predictions shape: {predictions.shape}")
        return predictions

    def decode_predictions(self, predictions):
        """Decode the predictions into readable text."""
        print("Decoding predictions...")
        decoded, _ = tf.keras.backend.ctc_decode(predictions, input_length=np.ones(predictions.shape[0]) * predictions.shape[1])
        texts = []
        for seq in decoded[0]:
            texts.append(''.join([self.char_list[c] for c in seq.numpy() if c != -1]))
        print("Decoding complete.")
        return texts

    def save(self, save_path="snapshot"):
        """Save the model to a file."""
        self.model.save(save_path)
        print(f"Model saved to {save_path}.")

    def load(self, load_path="snapshot"):
        """Load a saved model."""
        self.model = tf.keras.models.load_model(load_path, compile=False)
        print(f"Model loaded from {load_path}.")