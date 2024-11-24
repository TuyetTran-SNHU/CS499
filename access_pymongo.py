# Set up to connect to the mongodb 

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv


# Load environment variables from a .env file
load_dotenv()
# Retrieve the MongoDB URI from environment variables
uri = os.getenv('MONGODB_URI')

# Check if the URI exists
if uri:
    try:
        # Create a MongoClient instance with the URI and specify the MongoDB API version
        client = MongoClient(uri, server_api=ServerApi('1'))
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"An error occurred while connecting to MongoDB: {e}")
else:
    print("MongoDB URI not found. Please set the environment variable.")