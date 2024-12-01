import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Character list
char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


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

        def ctc_loss(y_true, y_pred):
            """Custom CTC loss function."""
            # Ensure y_true is integer and y_pred is float
            y_true = tf.cast(y_true, dtype=tf.int32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)

            tf.print("y_true shape (before):", tf.shape(y_true))

            # Compute input lengths (all sequences have the same time steps)
            input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])  # Shape: [batch_size]

            # Compute label lengths (count non-padded labels in y_true)
            label_length = tf.reduce_sum(tf.cast(y_true != -1, tf.int32), axis=1)  # Shape: [batch_size]

            # Debugging outputs
            tf.print("y_true shape:", tf.shape(y_true))
            tf.print("y_pred shape:", tf.shape(y_pred))
            tf.print("input_length shape:", tf.shape(input_length))
            tf.print("label_length shape:", tf.shape(label_length))

            # Compute the CTC loss
            return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=ctc_loss)
        print("Model compiled successfully.")

    def train_batch(self, batch_imgs, batch_labels, epochs=1):
        """Train the model with a batch of images and labels."""
        print(f"Starting training for {epochs} epochs...")

        # Validate shapes of inputs
        print("Validating shapes...")
        print("batch_imgs shape:", batch_imgs.shape)  # Should be [batch_size, height, width, channels]
        print("batch_labels:", batch_labels)         # Should be a list of strings
        for i, label in enumerate(batch_labels):
            print(f"Label {i}: '{label}' (length: {len(label)})")

        # Convert labels to sparse tensor
        sparse_labels = self.texts_to_sparse(batch_labels)
        
        # Convert sparse tensor to dense representation with padding (-1)
        dense_labels = tf.sparse.to_dense(sparse_labels, default_value=-1)
        
        # Further validation
        print("sparse_labels shape:", sparse_labels.dense_shape.numpy())  # Sparse tensor shape
        print("dense_labels shape:", dense_labels.shape)                 # Dense tensor shape
        
        # Train the model
        self.model.fit(x=batch_imgs, y=dense_labels, epochs=epochs, batch_size=batch_imgs.shape[0])
        print("Training complete.")

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

    def infer_batch(self, batch_imgs):
        """Run inference on a batch of images."""
        print("Running inference...")
        predictions = self.model.predict(batch_imgs)
        texts = self.decode_predictions(predictions)
        print("Inference complete.")
        return texts

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


if __name__ == "__main__":
    char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    model = Model(char_list=char_list)
    model.compile_model()
    
    # Dummy data
    batch_imgs = np.random.rand(2, 128, 32, 1)  # 2 images (Height=128, Width=32, Channels=1)
    batch_labels = ["words", "test"]  # Dummy labels

    model.train_batch(batch_imgs, batch_labels, epochs=1)
    predictions = model.infer_batch(batch_imgs)
    print(f"Inference results: {predictions}")
    model.save("snapshot")