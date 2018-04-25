import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from resnet_50 import resnet50_model

from os import path

from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH, IMG_HEIGHT = 227, 227
TRAIN_DATA = 'mart/standford-cars-crop/train'
VALID_DATA = 'mart/standford-cars-crop/valid'
NUM_CLASSES = 196
NB_TRAIN_SAMPLES = 6549
NB_VALID_SAMPLES = 1595
BATCH_SIZE = 40

# configure matplotlib
matplotlib.rc('axes', edgecolor='white')

# configurate the figure
plt.figure(figsize=(12, 12))

# enumerate the training data
categories = [category for category in os.listdir(TRAIN_DATA)]
numbers = [len(glob.glob(path.join(TRAIN_DATA, category, '*.jpg'))) for category in categories]

# plot the histogram
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(np.arange(len(categories)), numbers, width=0.5)
ax.set_title('histogram of training data')
plt.show()

# build a classifier model
model = resnet50_model(IMG_HEIGHT, IMG_WIDTH, 3, NUM_CLASSES)


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)
valid_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2)

train_generator = train_datagen.flow_from_directory(TRAIN_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')
valid_generator = valid_datagen.flow_from_directory(VALID_DATA, (IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')

# fine tune the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=NB_TRAIN_SAMPLES//BATCH_SIZE,
    validation_data=valid_generator,
    validation_steps=NB_VALID_SAMPLES//BATCH_SIZE,
    epochs=80)
