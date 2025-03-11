import os
import pandas as pd
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import blackwhite
from typing import List, Tuple


def split_csv(video_path: str, csv_path: str, saveflage: bool = False) -> List[Tuple[VideoFileClip, str]]:
    with VideoFileClip(video_path) as video_clip:
        timestamps = pd.read_csv(csv_path)

        videos = []

        # Iterate over timestamps to create clips
        for idx, row in timestamps.iterrows():
            start_time = row['start']
            end_time = row['end']
            word = row['word']

            # Extract clip within the specified start and end time
            clip = video_clip.subclip(start_time, end_time)
            
            #clip = clip.fx(blackwhite)

            videos.append((clip, word))

    return videos


def split_from_csv(input_dir="word_videos", output_dir="splitted_word_videos"):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop over each video in the directory
    for video_file in os.listdir(input_dir):
        if video_file.endswith((".mp4", ".avi", ".mov")):  # Adjust for your video format
            video_path = os.path.join(input_dir, video_file)
            csv_path = os.path.join(input_dir, os.path.splitext(video_file)[0] + ".csv")

            # Skip if no corresponding CSV file exists
            if not os.path.exists(csv_path):
                print(f"No CSV file found for {video_file}, skipping.")
                continue

            try:
                # Load video and timestamps
                with VideoFileClip(video_path) as video_clip:
                    timestamps = pd.read_csv(csv_path)

                    # Iterate over timestamps to create clips
                    for idx, row in timestamps.iterrows():
                        start_time = row['start']
                        end_time = row['end']
                        word = row['word']

                        # # Debug: Print the current word and timestamp
                        # print(f"Processing word: {word}, start: {start_time}, end: {end_time}")

                        # Create a folder for each unique word (if not already created)
                        word_folder = os.path.join(output_dir, word)
                        os.makedirs(word_folder, exist_ok=True)

                        
                        # Define clip filename in the format `[first unique 8 digits]_[word name].mp4`
                        clip_filename = f"{video_file[:8]}_{word}.mp4"
                        clip_output_path = os.path.join(word_folder, clip_filename)

                        # # Debug: Print the clip filename
                        # print(f"Saving clip: {clip_output_path}")

                        # Extract clip within the specified start and end time
                        clip = video_clip.subclip(start_time, end_time)

                        # Save the clip
                        clip.write_videofile(clip_output_path)

            except Exception as e:
                print(f"Error processing {video_file}: {e}")

    print(f"Splitting and saving {input_dir} into {output_dir} completed.")

# Example usage
# split_from_csv(input_dir=r"D:\Materials\Study\Graduation Project\lip-reading\New folder", output_dir=r"D:\Materials\Study\Graduation Project\lip-reading\split-output")

# #  D:/Materials/Study/Graduation Project/lip-reading/Datasets/01/LRW-AR/train/اتفاق/00003144_اتفاق.csv
# video_path = r"D:\Materials\Study\Graduation Project\lip-reading\Datasets\01\LRW-AR\train\الأمم\00004742_الأمم.mp4"
# csv_path = f"{video_path[:-3]}csv"
# videos = split_csv(video_path=video_path, csv_path=csv_path)
# print('----------------------------------------')
# (clip, word) = videos[0]
# print(clip)
# clip.write_videofile(f"Dataset/{word}.mp4", codec="libx264")