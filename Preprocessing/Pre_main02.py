import os
import pandas as pd
from moviepy.editor import VideoFileClip
from Preprocessing.HelperFunctions import *
from Preprocessing.StandardizeFPS import *
from Preprocessing.StandardizeDuration import *
from Preprocessing.Rotate import *
import sys
import io

FRAME_LEVEL_CROP = True

if FRAME_LEVEL_CROP:
    from Preprocessing.MouthBBox2 import *
    from Preprocessing.CropVids2 import *
else:
    from Preprocessing.MouthBBox import *
    from Preprocessing.CropVids import *

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def pre_main():
    # dataset_dir = getDirPath() # train data directory
    dataset_dir = r"D:\4th year\data from wageih\new we dataset"
    if not dataset_dir:
        return

    #target_words = ['كيف حالك','كيف اساعدك']
    target_user = ['User 22',]


    for User_name in os.listdir(dataset_dir):
        if User_name not in target_user:
            print('skip ', User_name)
            continue
        
        User_path = os.path.join(dataset_dir, User_name)
        print(User_path)

        for Word in os.listdir(User_path):
            # if Word not in target_user:
            #     print('skip ', user)
            #     continue

            Word_path = os.path.join(User_path, Word)
            # if not os.path.isdir(user_path):
            #     continue

            print(Word)
            for video_file in os.listdir(Word_path):
                print(video_file)
                if video_file.endswith((".mp4", ".avi", ".MOV")):  # Adjust for your video format
                    video_path = os.path.join(Word_path, video_file)
                    preprocess(video_path=video_path, word=Word, user = User_name)
                    
        
    



def preprocess(video_path, word,user):
    """
    1- standardize fps
    2- standardize duration
    3- get mouth region -> (Top-left, Buttom-right)
    4- croping the frames -> finalVideo
    """

    print("in preprocess from pre_main02")
    video_name = getVidName(video_path=video_path)
    print('Preprocessing ', video_name)

    with VideoFileClip(video_path) as clip:

            # 1- Rotate video if needed
            rotate_viedo_path = f"rotated_video.mp4"
            rotate(videoPath=video_path, outputPath=rotate_viedo_path)

            # 2 - Get mouth region boundaries
            x1, y1, x2, y2 = getMouthBBox(video_path=rotate_viedo_path, landmarks_path="")

            # 3 - perform cropping to get final video
            final_video_path = f"final_video.mp4"
            cropVideo(list(zip(x1, y1)), list(zip(x2, y2)), videoPath=rotate_viedo_path, outputPath=final_video_path)
            
           
            
        
# Example usage
# pre_main()
#preprocess(video_path=r"D:\4th year\data from wageih\new we dataset\User 17\اريد ماء\VID_20250308_235618.mp4", word="want water wigi",user="user17")



