import tensorflow as tf

def setup_rnn(cnn_output):
    """Build the RNN layers."""
    print("Setting up RNN layers...")
    x = tf.keras.layers.Reshape((-1, cnn_output.shape[-1]))(cnn_output)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    print("RNN setup complete.")
    return x
