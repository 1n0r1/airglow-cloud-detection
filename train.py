import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import PIL
import os


from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Read and split data
batch_size = 16
img_height = 519
img_width = 695
data_dir = 'labeled_splits/low_split'

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_dir, 
    batch_size=batch_size, 
    image_size=(img_height, img_width), 
    validation_split=0.2, 
    subset="both",
    seed=123)

class_names = train_ds.class_names
num_classes = len(class_names)


# Augment data
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                            input_shape=(img_height,
                                        img_width,
                                        3)),
        layers.RandomRotation(0.1)
    ]
)

# Specify model
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.CenterCrop(280, 280),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()


checkpoint_path = "training_checkpoints/cp-{epoch:04d}.tf"

# Checkpoint callback
mc = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1,
    save_freq='epoch')

# CSV callback to save history
cl = tf.keras.callbacks.CSVLogger('history.csv')

# Load checkpoint if want to continue from checkpoints
# model = tf.keras.models.load_model("old_checkpoints1/cp-0020.tf")

epochs=20
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    batch_size=1,
    callbacks=[cl, mc],
    verbose=1
)
