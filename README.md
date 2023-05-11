# Requirements

numpy 1.23.5

tensorflow 2.12.0

You need to have tensorflow installed. Refer to the official documentation and install tensorflow according to your system. If your system don't have a CUDA supported GPU, you might need to install tensorflow-cpu instead.

# Pulling images

modify fetch-images.py to set the site and the dates to pull from and the output folder

`python3 fetch-images.py` to generate cmd-list.sh

Then you can `bash cmd-list.sh` to start dowloading

# Get image list

`dir /data/blo/2022 >> text.txt`

# Get label

Put text.txt onto remote2 and run Cloud sensor data for ML training notebook. Besure to set the site in get_sub_temp_log. It will
read from text.txt and output is_cloudy.csv

# Sort raw data into folders

Download is_cloudy.csv to local machine and run sort_date_into_folders.ipynb

Be sure to change folder locations and blacklisat dates

# Train

Modify train.py to specify data folders and run `python3 train.py` or `python3 trainmixed.py` (that train data on multiple data)

It will output a folder of checkpoints for each epoch so you can continue training from an epoch without training from beginning over again

# Using the model

The trained model on both BLO and LOW is in trained_models/all40. A function that use it is in predict_cloudy.py, and you need to put
a single image in a seperate folder, for example:

main_directory/image_to_clasify.png

Then give the function the path to the folder.

As for why you need a seperate folder, it is because of a weird bug [here](https://stackoverflow.com/questions/72348377/why-does-image-dataset-from-directory-return-a-different-array-than-loading-imag) and [here](https://stackoverflow.com/questions/72348377/why-does-image-dataset-from-directory-return-a-different-array-than-loading-imag)

Try

`python predict_cloudy.py`

To check if the function can run or not