'''
https://blog.keras.io/building-autoencoders-in-keras.html
'''

'''
1. Load library and set default values
'''

# check GPU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.preprocessing.image import img_to_array
from matplotlib.pyplot import imshow
from tensorflow.keras.models import Sequential
import cv2
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Dropout
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

height = 700
width = 500

'''
2. Load training images  
'''


def create_training_data(data_path, height, width):
    training_data = []
    # for category in CATEGORIES:  # "baseline" and "rattle"

    #     path = os.path.join(data_path, category)  # create path
    #     # get the classification  (0 or a 1). 0=baseline 1=rattle
    #     class_index = CATEGORIES.index(category)

    # iterate over each image
    for image in os.listdir(data_path):
        # check file extention
        if image.endswith(".jpg"):
            try:
                data_path = pathlib.Path(data_path)
                full_name = str(pathlib.Path.joinpath(data_path, image))
                data = cv2.imread(str(full_name), 0)
                # resize to make sure data consistency
                resized_data = cv2.resize(data, (height, width))
                # add this to our training_data
                training_data.append([resized_data])
            except Exception as err:
                print("an error has occured: ", err, str(full_name))

    # normalize data
    training_data = np.array(training_data)/255.
    # reshape
    training_data = np.array(training_data).reshape(-1, height, width)
    return training_data


path = "D:/Data/Root/augmented"
data = create_training_data(path, height, width)
x_train = data[:-10]
x_test = data[-10:]


# fake data to check runtime errors
data = cv2.imread('root.jpg', 0)  # Change 1 to 0 for grey images
img_array = np.reshape(data, (-1, height, width, 1))
x_train = x_test = img_array.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train), height, width, 1))
x_test = np.reshape(x_test, (len(x_test), height, width, 1))


'''
3. Build autoencoder 
'''

input_img = keras.Input(shape=(height, width, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))


print("Neural network output")
pred = autoencoder.predict(img_array)


imshow(pred[0].reshape(height, width, 1), cmap="gray")
