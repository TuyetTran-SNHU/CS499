# CS499-Capstone Handwriting recognition

## Table of Content 

* [lmdb_setup](#lmbd_setup)
* [data_loader](#data_loader)
* [data_preprocessor](#data_preprocessor)
* [model](#model)
* [main](#main)
* [program](#program)


# lmdb_setup
## Database 
This file is designed to take a collection of images stored in a folder, convert them into a compact database format called LMDB, and store the images efficiently for future use. It first checks if the database (LMDB) already exists, then reads all .png images in a specified folder. Each image is converted to grayscale, encoded into PNG format, and saved into the LMDB using a unique identifier (the image file's name). This process helps in organizing and compressing image data, making it easier to manage and access for tasks like machine learning or large-scale data handling.

# data_loader
The purpose of this file is to define a program that helps organize and prepare image data for a ml model, specifically for recognizing text in images. It takes a folder of images and a file with information about each image (like its correct text) and splits the data into two groups: one for training the model and another for testing it. It can load images quickly from a database or directly from files, resize them to a standard size, and create "batches" of images and their correct text labels. This makes it easy to feed the data into a model in small, manageable chunks. Additionally, it includes tools to check if images are missing or can't be processed and logs these issues for troubleshooting.

# data_preprocessor 
The purpose of this file is to creates a "Preprocessor" tool that takes raw images of text and prepares them for use in a training and testing model. It resizes images to fit a standard size, adjusts their brightness, contrast, or sharpness, and applies random changes like blurring or noise to help the model learn better (this is called "data augmentation"). If necessary, it can also flip image colors and normalize pixel values to make processing easier. For text labels associated with images, it trims them to avoid issues during training. The preprocessor can handle batches of images at once, making it more efficient for large datasets. It's a helper program to clean up and modify images so the machine learning model can focus on learning patterns.

# model
The purpose of this file is to build a model for recognizing handwritten text from images using TensorFlow. It uses a combination of three components: a CNN (Convolutional Neural Network) to analyze image features, an RNN (Recurrent Neural Network) to understand the sequence of characters, and a CTC (Connectionist Temporal Classification) layer to map the model's output to readable text. The model is trained by feeding it batches of images and their corresponding text labels, improving its accuracy over time. It can also predict text from new images and provides functions for saving and loading the trained model for future use. This file ensures the model is well-structured, trains effectively, and is ready for deployment.


# main 
And lastly, this file is the main program that brings everything together to train and test a handwriting recognition model. It uses three key components: a DataLoader to organize and load image data, a Preprocessor to prepare and augment images for training, and a Model to learn how to recognize handwritten text. It first checks if a database (LMDB) of preprocessed images exists, creating it if necessary. The model is trained using the data, and its performance is evaluated on unseen validation images. After training, the model is saved and tested to see how well it predicts the text from the input images. The program ensures the whole process runs smoothly, from data preparation to testing the final trained model.

# program
This extra file is to create a simple handwriting recognition app with a graphical user interface (GUI). It allows users to upload an image of handwritten text, processes the image, and predicts the text using a trained machine learning model. The program uses Tkinter for the GUI, allowing users to select an image file from their computer. The image is displayed in the app, processed by a preprocessor to prepare it for the model, and then the model predicts the text. The result is displayed on the screen. This app combines machine learning and user-friendly design to make handwriting recognition accessible and interactive.
















# CS499-Capstone
