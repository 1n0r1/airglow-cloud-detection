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

Modify train.py to specify data folders and run `python3 train.py`

It will output a folder of checkpoints for each epoch so you can continue training from an epoch without training from beginning over again