from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

# Define and compile the model
def create_model():
    # Model type is Sequential 
    model = Sequential()
    # add dense to the model
    model.add(Dense(10, input_shape=(784,)))
    # Add activation softmax alogrith
    model.add(Activation('softmax'))
    # Add activation relu alogrithm
    model.add(Activation('relu'))
    # assign the type of opimizer
    optimizer = Adam(learning_rate=0.01, momentum=0.9)
    # compile the alogrithm and calculate the loss and accuracy
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model