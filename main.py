from data_preprocessing import load_and_preprocess_data
from model_definition import create_model
from training import train_and_evaluate
from user_verification import user 
from image_input import upload_image
import sys

#user variable to user in the image database 
username = user()
if not username:  
    # if user() returns False then verification fails
    print("User verification failed. Exiting the program.")
    # Exit the program if user verification fails
    sys.exit() 

# Load and preprocess data
X_train, Y_train, X_test, Y_test = load_and_preprocess_data()

# Define and compile the model
model = create_model()
model.summary()
1
# Train and evaluate the model
score = train_and_evaluate(model, X_train, Y_train, X_test, Y_test)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])

# image_input file 
image_path = input("Enter the path to the image: ")
# Upload the image and user info to MongoDB
upload_image(image_path, username)

