def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, batch_size=128, epochs=200, validation_split=0.2):
    # Train the model with training data, and validate with a split of the data
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)
    
    # Evaluate the trained model on the test data and return the score (loss and accuracy)
    score = model.evaluate(X_test, Y_test, verbose=1)
    
    # Return the evaluation results
    return score