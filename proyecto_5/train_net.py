import cv2
import numpy as np
import os
from os import path
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers, regularizers


if path.isdir('image_data') and path.isdir('image_test'):
    TRAINING_DIR = "image_data"
    TEST_DIR = "image_test"
    trainig_datagen = ImageDataGenerator(rescale = 1./255)
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = trainig_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size = (150,150),
        class_mode = 'categorical'
    )

    val_generator = validation_datagen.flow_from_directory(
        TEST_DIR,
        target_size = (150,150),
        class_mode = 'categorical'
    )


    model = models.Sequential([
        layers.Conv2D(64, (3,3), activation='relu', input_shape = (150,150,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dropout(0.5),

        layers.Dense(512, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])

    model.compile(loss= 'categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy']
    )

    history = model.fit(train_generator, epochs=9, 
                    validation_data=val_generator,
                    verbose= 1
                    )

    model.save("rock-paper-scissors-model.h5")
else:
    print("No hay datos")