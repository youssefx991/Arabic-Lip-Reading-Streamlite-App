
import streamlit as st
import os 
import imageio

import tensorflow as tf
from utils import *
from modelutil import *
from Preprocessing.Pre_main02 import preprocess

st.set_page_config(layout='wide')


with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App') 

# File uploader for manual video selection
uploaded_file = st.file_uploader("Upload a video")


# Generate two columns 
col1, col2 = st.columns(2)


if uploaded_file is not None:
    # # Define directory to save uploaded files
    save_dir = "uploaded_videos"
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    # # Create full file path
    file_path = os.path.join(save_dir, uploaded_file.name)
    abs_file_path = os.path.abspath(file_path)

    os.system(f'ffmpeg -i {abs_file_path} -vcodec libx264 test_video.mp4 -y')

    # Save the uploaded file
    with open('test_video.mp4', "wb") as f:
        f.write(uploaded_file.read())

    preprocess(video_path='test_video.mp4', word='test', user='test')

    # # Generate two columns 
    col1, col2 = st.columns(2)

    # # Rendering the video 
    with col1:
        st.info('The video below displays the uploaded video:')
        st.video('test_video.mp4')  # Correct way to display the video

    # # Show full path (for debugging)
    # st.write(f"File saved at: `{abs_file_path}`")
    # st.write(f"abs path for video: {os.path.abspath(file_path)}")

    with col2: 
        def check_file_exists(file_path):
            if os.path.exists(file_path):
                st.success(f"File {file_path} exists.")
            else:
                st.error(f"File {file_path} does not exist.")

        check_file_exists('cropped_video.mp4')
        # st.info('This is all the machine learning model sees when making a prediction')
        video, annotations = load_new_data(tf.convert_to_tensor('cropped_video.mp4'))
        # imageio.mimsave('animation.gif', video, fps=10)
        # st.image('animation.gif', width=400) 

        # st.info('This is the output of the machine learning model as tokens')
        model = create_model()
        yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)
        decoder = tf.keras.backend.ctc_decode(yhat, [60], greedy=True)[0][0].numpy()
        # st.text(decoder)

        # Convert prediction to text
        st.info('This is what the model predicts to be said in video')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
