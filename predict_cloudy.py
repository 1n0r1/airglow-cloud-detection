import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import PIL
import os

import numpy as np


def classify(img_folder_path):
    """Take a folder path containing an image to predict and return the prediction and confidence level
    """
    img_height = 519
    img_width = 695
    print(img_folder_path)
    img = keras.utils.image_dataset_from_directory(img_folder_path, image_size=(img_height, img_width), label_mode = None)

    model = tf.keras.models.load_model("trained_models/all40")
    for x in img:
        prediction = model(x)
        score = tf.nn.softmax(prediction[0])
        confidence = 100 * np.max(score) 
        if prediction[0, 0] >= prediction[0,1]:
            return 'Clear', confidence
        else:
            return 'Cloudy', confidence
        
if __name__ == "__main__":
    res1 = classify('test_image/low_clear')
    res2 = classify('test_image/low_cloudy')
    res3 = classify('test_image/cvo_clear')
    res4 = classify('test_image/cvo_cloudy')
    res5 = classify('test_image/blo_clear')
    res6 = classify('test_image/blo_cloudy')
    print(res1)
    print(res2)
    print(res3)
    print(res4)
    print(res5)
    print(res6)
