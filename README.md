# CS499-Capstone Handwriting recognition

## Table of Content 

* [lmdb_setup](#lmbd_setup)
* [data_loader](#data_loader)
* [data_preprocessor](#data_preprocessor)
* [model](#model)
* [main](#main)


# lmdb_setup
## Database 
This file is designed to take a collection of images stored in a folder, convert them into a compact database format called LMDB, and store the images efficiently for future use. It first checks if the database (LMDB) already exists, then reads all .png images in a specified folder. Each image is converted to grayscale, encoded into PNG format, and saved into the LMDB using a unique identifier (the image file's name). This process helps in organizing and compressing image data, making it easier to manage and access for tasks like machine learning or large-scale data handling.

# data_loader






















# CS499-Capstone
