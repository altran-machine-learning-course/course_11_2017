import joblib
x_train_minimal, y_train_minimal = joblib.load("traindata_minimal.pkl")
x_test_minimal, y_test_minimal = joblib.load("testdata_minimal.pkl")
x_test, y_test = joblib.load("testdata.pkl") #scaled final data to test


# Lets check out a basic random forest before we go far
from sklearn.ensemble import RandomForestClassifier as RFC
classifier = RFC(n_estimators=100)
classifier.fit(x_train_minimal, y_train_minimal)
train_acc = classifier.score(x_train_minimal, y_train_minimal) # training score
print("Training accuracy is : "+str(train_acc))
import sys
sys.path.insert(0,'../input')
from utils import accuracy_score_numpy
test_accuracy = accuracy_score_numpy(classifier.predict(x_test))
print ("Test accuracy is : "+str(test_accuracy))


## Preparing for Keras
import random
def get_random_batch(n_samples=10):
    selection_list = random.sample(range(0,len(x_train_minimal)), n_samples)
    return x_train_minimal[selection_list], y_train_minimal[selection_list]



#Real Keras starts here

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import numpy as np


import sys

import numpy as np
NOISE_SHAPE = 50
class DCGAN():
    def __init__(self):
        # self.img_rows = 28
        # self.img_cols = 28
        self.data_rows = 1
        self.data_cols = 11
        self.channels = 1

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator() # Some Magical function
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator() # Another Magical Function
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(NOISE_SHAPE,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid) # Lets connect the two models
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

def build_generator(self):

    noise_shape = (NOISE_SHAPE,)

    model = Sequential()

    model.add(Dense(20 * 11, activation="relu", input_shape=noise_shape))
    model.add(Dense(30 * 11, activation="relu", input_shape=noise_shape))
    model.add(Dense(50 * 11, activation="relu", input_shape=noise_shape))
    model.add(Dense(30 * 11, activation="relu", input_shape=noise_shape))
    model.add(Dense(1 * 11, activation="relu", input_shape=noise_shape))
    # model.add(Reshape((7, 7, 128)))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(UpSampling2D())
    # model.add(Conv2D(128, kernel_size=3, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(UpSampling2D())
    # model.add(Conv2D(64, kernel_size=3, padding="same"))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Conv2D(1, kernel_size=3, padding="same"))
    # model.add(Activation("tanh"))
    model.summary()

    noise = Input(shape=noise_shape)
    gen_data_sample = model(noise)

    return Model(noise, gen_data_sample)
DCGAN.build_generator = build_generator



def build_discriminator(self):

    img_shape = (self.data_cols)

    model = Sequential()

    model.add(Dense(20 * 11, activation="relu", input_shape=img_shape))
    model.add(Dense(30*11), activation="relu", input_shape=img_shape)
    model.add(Dense(40*11), activation="relu", input_shape=img_shape)
    model.add(Dense(30*11), activation="relu", input_shape=img_shape)
    model.add(Dense(10*11), activation="relu", input_shape=img_shape)
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

DCGAN.build_discriminator = build_discriminator