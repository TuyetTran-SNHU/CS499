# import mnist dataset 
from keras.datasets import mnist 
# import labels and format
from keras.utils import to_categorical  

# Load and preprocess the data
def load_and_preprocess_data():
    # Load MNIST data, with training and testing sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Reshape and normalize training data (flattening images to 784 pixels and scaling to [0,1] range)
    X_train = X_train.reshape(60000, 784).astype('float32') / 255
    
    # Reshape and normalize test data in the same way as training data
    X_test = X_test.reshape(10000, 784).astype('float32') / 255
    
    # Convert labels to one-hot encoded format for 10 classes (0-9 digits) for training data
    Y_train = to_categorical(y_train, 10)
    
    # Convert labels to one-hot encoded format for test data
    Y_test = to_categorical(y_test, 10)
    
    # Return preprocessed training and test sets
    return X_train, Y_train, X_test, Y_test