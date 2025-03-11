import os
import pandas as pd
from moviepy.editor import VideoFileClip
from HelperFunctions import *
from split import split_csv
from StandardizeFPS import *
from StandardizeDuration import *
from MouthBBox import *
from CropVids import *
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def pre_main():
    dataset_dir = getDirPath() # train data directory
    if not dataset_dir:
        return

    #target_words = ['اتفاق', 'الإسرائيلي', 'الإسلامية','الأمم','الأمن']
    target_words = ['السلام','عليكم','الله','اتفاق','البلاد','الجيش']

    for folder_name in os.listdir(dataset_dir):
        if folder_name not in target_words:
            continue
        
        folder_path = os.path.join(dataset_dir, folder_name)
        print(folder_path)
        for video_file in os.listdir(folder_path):
            if video_file.endswith((".mp4", ".avi", ".mov")):  # Adjust for your video format
                video_path = os.path.join(folder_path, video_file)
                csv_path = csv_path = f"{video_path[:-3]}csv"

                # Skip if no corresponding CSV file exists
                if not os.path.exists(csv_path):
                    print(f"No CSV file found for {video_file}, skipping.")
                else:
                    preprocess(video_path=video_path, csv_path=csv_path)


def preprocess(video_path, csv_path):
    """
    1- orginalVideo -> split using csv -> videos
    2- standardize fps
    3- standardize duration
    4- get mouth region -> (Top-left, Buttom-right)
    5- croping the frames -> finalVideo
    """

    # # 1- slit using csv
    # splitted_videos = split_csv(video_path=video_path, csv_path=csv_path)  #(clip, word)

    video_name = getVidName(video_path=video_path)

    with VideoFileClip(video_path) as video_clip:
        # 1- slit using csv
        timestamps = pd.read_csv(csv_path)

        # Iterate over timestamps to create clips
        for idx, row in timestamps.iterrows():
            start_time = row['start']
            end_time = row['end']
            word = row['word']

            # Extract clip within the specified start and end time
            clip = video_clip.subclip(start_time, end_time)

            # 2- standardize fps
            video_with_new_fps = standardizeFPS(video=clip, targetFPS=25)

            # 3- standardize duration
            video_stretched = standardizeDuration(video=video_with_new_fps, targetDuration=1)

            # Save the standardized video
            dir_path = f"./Dataset/{word}"
            dirExists(dir_path=dir_path) # check if the dir exists, if not creat it
            save_path = f"{dir_path}/{video_name}({word}).mp4" 
            video_stretched.write_videofile(save_path, codec="libx264")

            # 4- get mouth region
            landmarks_path = f"{save_path[:-4]}_lm.txt"
            x1, y1, x2, y2 = getMouthBBox(video_path=save_path, landmarks_path=landmarks_path)

            # 5- croping the frames and save
            cropped_video_path = f"{save_path[:-4]}_cropped.mp4"
            cropVideo((x1,y1), (x2,y2), videoPath=save_path, outputPath=cropped_video_path)
        

# Example usage
# video_path = r"D:\Materials\Study\Graduation Project\lip-reading\Datasets\01\LRW-AR\train\الأمم\00004742_الأمم.mp4"

### run preprocessing on data set 

# video_path = getFilePath()
# csv_path = f"{video_path[:-3]}csv"
# preprocess(video_path=video_path, csv_path=csv_path)


pre_main()