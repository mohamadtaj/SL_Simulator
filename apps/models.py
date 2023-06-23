import tensorflow as tf
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding, LSTM
from keras.layers import GlobalAveragePooling1D
from tensorflow.keras import datasets, models, Model, Input
from tensorflow.keras import regularizers
       


def model(input_size, output_size):

    initializer = tf.keras.initializers.he_uniform()    
    model = models.Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', kernel_initializer=initializer, padding='same', input_shape = input_size))
    model.add(Conv1D(32, kernel_size=3, activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, kernel_size=3, activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(Conv1D(64, kernel_size=3, activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(Conv1D(128, kernel_size=3, activation='relu', kernel_initializer=initializer, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=initializer))
    model.add(Dropout(0.5))     
    model.add(Dense(output_size, activation='sigmoid'))
    
    return model