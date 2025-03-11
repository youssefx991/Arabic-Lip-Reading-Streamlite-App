import os
import pandas as pd
from moviepy.editor import VideoFileClip
from Preprocessing.HelperFunctions import *
from Preprocessing.StandardizeFPS import *
from Preprocessing.StandardizeDuration import *
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

# def pre_main():
#     # dataset_dir = getDirPath() # train data directory
#     dataset_dir = r"D:\4th year\data from wageih\we Dataset for training"
#     if not dataset_dir:
#         return

#     #target_words = ['كيف حالك','كيف اساعدك']
#     target_user = ['User6',]


#     for folder_name in os.listdir(dataset_dir):
#         # if folder_name not in target_words:
#         #     print('skip ', folder_name)
#         #     continue
        
#         word_path = os.path.join(dataset_dir, folder_name)
#         print(word_path)

#         for user in os.listdir(word_path):
#             if user not in target_user:
#                 print('skip ', user)
#                 continue

#             user_path = os.path.join(word_path, user)
#             # if not os.path.isdir(user_path):
#             #     continue

#             print(user)
#             for video_file in os.listdir(user_path):
#                 print(video_file)
#                 if video_file.endswith((".mp4", ".avi", ".mov")):  # Adjust for your video format
#                     video_path = os.path.join(user_path, video_file)
#                     preprocess(video_path=video_path, word=folder_name, user = user)
#                     #return
        
    



def preprocess(video_path, word,user):
    """
    1- standardize fps
    2- standardize duration
    3- get mouth region -> (Top-left, Buttom-right)
    4- croping the frames -> finalVideo
    """

    video_name = getVidName(video_path=video_path)
    print('Preprocessing ', video_name)

    with VideoFileClip(video_path) as clip:

            # 1- standardize fps

            # 2- standardize duration

            # # Save the standardized video
            # dir_path = f"./Dataset/{word}"
            # dirExists(dir_path=dir_path) # check if the dir exists, if not creat it
            # save_path = f"{dir_path}/{video_name}({word}).mp4" 
            # video_stretched.write_videofile(save_path, codec="libx264")

            # 4- get mouth region
            landmarks_path = f"{video_path[:-4]}_lm.txt"
            x1, y1, x2, y2 = getMouthBBox(video_path=video_path, landmarks_path=landmarks_path)

            # 5- croping the frames and save
            dir_path = f"./Dataset/{word}/{user}"
            dirExists(dir_path=dir_path)
            save_path = f"{dir_path}/{video_name}.mp4" 
            cropped_video_path = f"{save_path[:-4]}_cropped.mp4"

            # if FRAME_LEVEL_CROP: 
            cropVideo(list(zip(x1, y1)), list(zip(x2, y2)), videoPath=video_path, outputPath=cropped_video_path)
            # else:
            #     cropVideo((x1,y1), (x2,y2), videoPath=video_path, outputPath=cropped_video_path)
           
            
        
# Example usage
#pre_main()
# preprocess(video_path=r"C:\Users\max\Desktop\اتقن اللغة العربية\فيديوهاتي\اريد ماء\VID20250308212313.mp4", word="want water",user="user19")

