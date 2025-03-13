import cv2
import imageio
import numpy as np
import os
import tensorflow as tf

from typing import List,Tuple
from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, BatchNormalization, Bidirectional, Conv3D, Conv2D, Dense, Dropout, Flatten,
                                     GRU, LSTM, MaxPool3D, Reshape, SpatialDropout3D, TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.legacy import Adam


import cv2
import imageio
from tensorflow.keras.layers import Conv3D, ZeroPadding3D, MaxPooling3D, Dense, Activation, Dropout, Flatten, Input
from tensorflow.keras.models import Model
import random
import shutil
import glob

from utils import *
target_fps = 60


def create_model():
    input_data = Input(name='the_input', shape=(target_fps,150,160,1), dtype='float32')
    zero1 = ZeroPadding3D(padding=(1, 2, 2), name='zero1')(input_data)
    conv1 = Conv3D(32, (3, 5, 5), strides=(1, 2, 2), kernel_initializer='he_normal', name='conv1')(zero1)
    batc1 = BatchNormalization(name='batc1')(conv1)
    actv1 = Activation('relu', name='actv1')(batc1)
    drop1 = SpatialDropout3D(0.5)(actv1)
    maxp1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max1')(drop1)

    zero2 = ZeroPadding3D(padding=(1, 2, 2), name='zero2')(maxp1)
    conv2 = Conv3D(64, (3, 5, 5), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv2')(zero2)
    batc2 = BatchNormalization(name='batc2')(conv2)
    actv2 = Activation('relu', name='actv2')(batc2)
    drop2 = SpatialDropout3D(0.5)(actv2)
    maxp2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max2')(drop2)

    zero3 = ZeroPadding3D(padding=(1, 1, 1), name='zero3')(maxp2)
    conv3 = Conv3D(96, (3, 3, 3), strides=(1, 1, 1), kernel_initializer='he_normal', name='conv3')(zero3)
    batc3 = BatchNormalization(name='batc3')(conv3)
    actv3 = Activation('relu', name='actv3')(batc3)
    drop3 = SpatialDropout3D(0.5)(actv3)
    maxp3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name='max3')(drop3)

    resh1 = TimeDistributed(Flatten())(maxp3)

    gru_1 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru1'), merge_mode='concat')(resh1)
    gru_2 = Bidirectional(GRU(256, return_sequences=True, kernel_initializer='Orthogonal', name='gru2'), merge_mode='concat')(gru_1)

    # transforms RNN output to character activations:
    dense1 = Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', name='dense1')(gru_2)

    y_pred = Activation('softmax', name='softmax')(dense1)

    model = Model(inputs=input_data, outputs=y_pred)

    model.load_weights('./checkpoint_our_dataset_43_epoch_01.h5')
    return model

model = create_model()
