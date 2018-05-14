from __future__ import division, print_function, absolute_import
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
import numpy as np
from hw4 import dataparser as dp
# import pdb

# Define Model
def cnnmodel(dim, classes):
    model = Sequential()
    # First Convolution Layer
    model.add(Conv3D(32, kernel_size = (5,5,5), strides = (1,1,1), padding = 'same', 
              activation = 'relu', input_shape=(dim, dim, dim, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    # Second Convolution Layer
    model.add(Conv3D(64, kernel_size = (3,3,3), strides = (1,1,1), 
              activation = 'relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model

# Create input Data
def preprocess_data(path_data, d):
    (X_train, Y_train), (X_test, Y_test), classes = dp.create_input(path_data, d)
    x_train = np.asarray(X_train)
    x_test = np.asarray(X_test)
    y_train = np.asarray(Y_train)
    y_test = np.asarray(Y_test)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(x_train.shape[0], d, d,d, 1)
    x_test = x_test.reshape(x_test.shape[0], d, d, d, 1)
    y_train = keras.utils.to_categorical(y_train, classes)
    y_test = keras.utils.to_categorical(y_test, classes)
    return (x_train, y_train), (x_test, y_test), classes

# Train the model
def trainmodel(path_data, d):   
    # Hyper Parameters
    batch_size = 60
    epochs = 500
    #Preprocessing
    (x_train, y_train), (x_test, y_test), classes = preprocess_data(path_data, d)
    # Create Model
    model = cnnmodel(d, classes)
    # Training Starts
    print("##############  Training Starts  ##############")
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs = epochs, verbose=1, validation_data=(x_test,y_test))
    score = model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss', score[0])
    print('Test accuracy:', score[1])
