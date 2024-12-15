from PIL import Image
from pymongo import MongoClient
import io
import base64
import access_pymongo

# Connect to the mongoDB 
client = access_pymongo.client

# Database and collection references
db = client["db"]  
collection = db["images"]

# Modular function for uploading image and preprocessing
def upload_image(image_path, username):
    try:
        # Load the image
        with Image.open(image_path) as img:
            # Resize the image to 28x28 pixels to match MNIST requirements and testing 
            img = img.resize((28, 28))

            # Convert the image to grayscale if it isn't already
            img = img.convert("L")

            # Convert the resized image to bytes
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format="PNG")
            img_data = img_byte_array.getvalue()

            # Encode the image to base64
            img_base64 = base64.b64encode(img_data).decode("utf-8")
            
            # Prepare the document for MongoDB
            document = {
                "username": username,
                "image_data": img_base64,
                "image_format": "PNG"
            }
            
            # Insert into MongoDB
            result = collection.insert_one(document)
            print(f"Image uploaded successfully with ID: {result.inserted_id}")
    
    # Error handeling 
    except FileNotFoundError:
        print("Error: The image file was not found. Please check the path and try again.")
    except Exception as e:
        print(f"An error occurred during image upload: {e}")
