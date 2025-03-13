import cv2
import imageio
import numpy as np
import os
import tensorflow as tf
import streamlit as st

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

target_fps = 60


vocab = ['{}'.format(x) for x in " اأبتثجحخدذرزسشصضطظعغفقكلمنهـويةءىئ"]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
# print(f"The vocabulary is: {char_to_num.get_vocabulary()}")
# print(f"(size ={char_to_num.vocabulary_size()})")


def convert_digits_to_arabic(phrase: str) -> str:
    # Map each digit in the phrase to its Arabic word
    digit_to_arabic = {
    '0': 'صفر',  # sifr
    '1': 'واحد',  # wahid
    '2': 'اثنان',  # ithnan
    '3': 'ثلاثة',  # thalatha
    '4': 'اربعة',  # arba’a
    '5': 'خمسة',   # khamsa
    '6': 'ستة',    # sitta
    '7': 'سبعة',   # sab’a
    '8': 'ثمانية', # thamaniya
    '9': 'تسعة'    # tis’a
}


    # Replace digits with their Arabic equivalents
    arabic_phrase = ''.join(digit_to_arabic.get(char, char) for char in phrase)
    return arabic_phrase

# # Example usage:
# original_phrase = "اتصل بالشرطة"
# arabic_phrase = convert_digits_to_arabic(original_phrase)
# print(f"Original phrase: {original_phrase}")
# print(f"Arabic phrase: {arabic_phrase}")

def load_video(path:str) -> List[float]:
    cap = cv2.VideoCapture(path)

    check_file_exists(path)
    st.info(f"int(cap.get(cv2.CAP_PROP_FRAME_COUNT)): {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    if not cap.isOpened():
        st.error(f"Error: Could not read video file {path}")
        return
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (160, 150))  # Resize to 160x150
        frame = tf.image.rgb_to_grayscale(frame)
        frame=frame/255
        # frames.append(frame[:150,70:230,:])
        frames.append(frame)
    cap.release()

    # target_fps = 30
    if len(frames) < target_fps:
        padding = target_fps - len(frames)
        # Repeat last frame to pad instead of using zeros
        frames += [frames[-1]] * padding
    if len(frames) > target_fps:
        # Downsampling to exactly 60 frames
        frames = frames[:target_fps]  # Select the first 60 frames

    # mean = tf.math.reduce_mean(frames)
    # std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return  tf.cast((frames), tf.float32)


def load_alignments(phrase:str) -> List[str]:
    tokens = []
    arabic_phrase = convert_digits_to_arabic(phrase)
    line = arabic_phrase.split()
    for l in line:
        tokens = [*tokens,' ',l]
    res = ''.join(tokens)

    # print(f"Original phrase: {res}")  # Debug output
    # print(f"arabic phrase: {arabic_phrase}")  # Debug output
    alignment=char_to_num(tf.reshape(tf.strings.unicode_split(res, input_encoding='UTF-8'), (-1)))[1:]
    # print(f"Original alignment (before padding): {alignment.numpy()}")  # Debug output


    return alignment

def load_new_data(path: str):

    video_path = bytes.decode(path.numpy())

    # print("Extracting frames & alignments from new video")
    # print(f"Current Path: {video_path}")
    frames = load_video(video_path)
    alignment = load_alignments('فيديو جديد')
    return frames, alignment

def check_file_exists(file_path):
    if os.path.exists(file_path):
        pass
        # st.success(f"File {file_path} exists.")
    else:
        st.error(f"File {file_path} does not exist.")