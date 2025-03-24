
import streamlit as st
import os 
import imageio

import tensorflow as tf
from utils import *
from modelutil import *
from Preprocessing.Pre_main02 import preprocess
import os

def main():
    st.set_page_config(layout='wide')
    
    # download_weights()
    with st.sidebar:
        st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
        st.title('Arabic Lip Reading')
        st.info('This application is originally developed from the LipNet deep learning model.')

    st.title('Arabic Lip Reading')

    uploaded_file = st.file_uploader("Upload a video")
    preprocess_option = st.checkbox('Crop Mouth Region video before analysis')

    if uploaded_file is not None:
        process_uploaded_file(uploaded_file, preprocess_option)

def process_uploaded_file(uploaded_file, preprocess_option):
    # save_dir = "uploaded_videos"
    # os.makedirs(save_dir, exist_ok=True)
    # file_path = os.path.join(save_dir, uploaded_file.name)
    # abs_file_path = os.path.abspath(file_path)
    abs_file_path = os.path.abspath(uploaded_file.name)

    os.system(f'ffmpeg -i {abs_file_path} -vcodec libx264 test_video.mp4 -y')

    with open('test_video.mp4', "wb") as f:
        f.write(uploaded_file.read())

    video_name = 'test_video.mp4'
    if preprocess_option:
        try:
            preprocess(video_path='test_video.mp4', word='test', user='test')
            video_name = 'final_video.mp4'
            abs_video_path = os.path.abspath(video_name)

            video_frames = imageio.mimread('final_video.mp4', memtest=False)
            imageio.mimsave('final_video.gif', video_frames, fps=10, loop=0)
            os.system(f'ffmpeg -i {abs_video_path} -vcodec libx264 {video_name} -y')
            st.image('final_video.gif', width=400)
            with open('final_video.mp4', 'rb') as f:
                st.download_button('Download Cropped Video', f, file_name='final_video.mp4')
        except Exception as e:
            st.error("Error during preprocessing, will continue without cropping")
            video_name = 'test_video.mp4'

    display_video_and_analyze(video_name)

def display_video_and_analyze(video_name):
    col1, col2 = st.columns(2)

    with col1:
        st.info('The video below displays the uploaded video:')
        st.video("test_video.mp4")

    with col2:
        check_file_exists(video_name)
        if st.button('Analyze Video'):
            with st.spinner('Analyzing video...'):
                try:
                    abs_video_path = os.path.abspath(video_name)
                    # st.write(f"Absolute path of video: {abs_video_path}")
                    video, annotations = load_new_data(tf.convert_to_tensor(abs_video_path))
                    
                    model = create_model()
                    yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)
                    decoder = tf.keras.backend.ctc_decode(yhat, [60], greedy=True)[0][0].numpy()

                    st.info('This is what the model predicts to be said in video')
                    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
                    st.text(converted_prediction)
                except Exception as e:
                    st.error("Error during video analysis, please make sure you uploaded a valid video file and selected the correct cropping option")

    # display_files_in_directory()

def display_files_in_directory():
    files = os.listdir('.')
    st.write("Files in the directory:")
    for file in files:
        st.write(file)
        # pass

if __name__ == "__main__":
    main()

# st.set_page_config(layout='wide')


# with st.sidebar: 
#     st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
#     st.title('LipBuddy')
#     st.info('This application is originally developed from the LipNet deep learning model.')

# st.title('LipNet Full Stack App') 

# # File uploader for manual video selection
# uploaded_file = st.file_uploader("Upload a video")

# preprocess_option = st.checkbox('Crop Mouth Region video before analysis')


# # Generate two columns 
# col1, col2 = st.columns(2)


# if uploaded_file is not None:
#     # # Define directory to save uploaded files
#     save_dir = "uploaded_videos"
#     os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

#     # # Create full file path
#     file_path = os.path.join(save_dir, uploaded_file.name)
#     abs_file_path = os.path.abspath(file_path)

#     os.system(f'ffmpeg -i {abs_file_path} -vcodec libx264 test_video.mp4 -y')

#     # Save the uploaded file
#     with open('test_video.mp4', "wb") as f:
#         f.write(uploaded_file.read())

#     # Checkbox for preprocessing option

#     video_name = 'test_video.mp4'
#     if preprocess_option:
#         try:
#             preprocess(video_path='test_video.mp4', word='test', user='test')
#             video_name = 'final_video.mp4'
#         except Exception as e:
#             st.error(f"Error during preprocessing, will continue without cropping")
#             video_name = 'test_video.mp4'    
#     else:
#         video_name = 'test_video.mp4'

#     # # Generate two columns 
#     col1, col2 = st.columns(2)

#     # # Rendering the video 
#     with col1:
#         st.info('The video below displays the uploaded video:')
#         st.video("test_video.mp4")  # Correct way to display the video

#     # # Show full path (for debugging)
#     # st.write(f"File saved at: `{abs_file_path}`")
#     # st.write(f"abs path for video: {os.path.abspath(file_path)}")
#     with col2: 
#         check_file_exists(video_name)
#         # Button to trigger model prediction
#         if st.button('Analyze Video'):
#             with st.spinner('Analyzing video...'):
#                 #st.info('This is all the machine learning model sees when making a prediction')
#                 try:
#                     abs_video_path = os.path.abspath(video_name)
#                     st.write(f"Absolute path of video: {abs_video_path}")
#                     video, annotations = load_new_data(tf.convert_to_tensor(abs_video_path))
#                     # imageio.mimsave('animation.gif', video, fps=10)
#                     # st.image('animation.gif', width=400) 

#                     # st.info('This is the output of the machine learning model as tokens')
#                     model = create_model()
#                     yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)
#                     decoder = tf.keras.backend.ctc_decode(yhat, [60], greedy=True)[0][0].numpy()
#                     # st.text(decoder)

#                     # Convert prediction to text
#                     st.info('This is what the model predicts to be said in video')
#                     converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
#                     st.text(converted_prediction)
#                 except Exception as e:
#                     st.error(f"Error during video analysis, please make sure you uploaded a valid video file and seleceted the correct cropping option")
        


#     # Get the list of files in the current directory
#     files = os.listdir('.')

#     # Display the list of files
#     st.write("Files in the directory:")
#     for file in files:
#         st.write(file)