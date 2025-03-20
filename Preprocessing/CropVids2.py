import cv2
import os
import numpy as np

def cropVideo(topLeftPoints, bottomRightPoints, videoPath, outputPath="cropped_video.mp4"):
    """
    Crop each frame of the video using per-frame top-left and bottom-right coordinates.
    The cropped region of each frame is converted to grayscale, resized to a fixed 
    resolution (160x150), and written to an output video file at 30 FPS.

    Args:
        topLeftPoints (list of (int,int)): List of top-left corner coordinates per frame.
        bottomRightPoints (list of (int,int)): List of bottom-right corner coordinates per frame.
        videoPath (str): Path to the input video.
        outputPath (str): Path to save the cropped video.
    """
    print("in cropVideo from CropVids2")
    # Load the video
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error opening video file:", videoPath)
        return

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps  # Preserve duration

    # Define the fixed output size for the cropped video
    fixed_width, fixed_height = 160, 150
    target_fps = 30.0  # Set output FPS

    # Compute the frame sampling ratio
    frame_interval = original_fps / target_fps if original_fps > target_fps else target_fps / original_fps

    # Define the output video writer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(outputPath, fourcc, target_fps, (fixed_width, fixed_height), isColor=False)

    frame_index = 0
    output_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Determine if this frame should be processed (frame skipping or duplication)
        process_frame = (original_fps > target_fps and frame_index % round(frame_interval) == 0) or (original_fps <= target_fps)

        if process_frame:
            # Get cropping coordinates
            TL = topLeftPoints[min(frame_index, len(topLeftPoints) - 1)]
            BR = bottomRightPoints[min(frame_index, len(bottomRightPoints) - 1)]
            x1, y1 = TL[0], TL[1]
            x2, y2 = BR[0], BR[1]

            # Calculate width and height of the region
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            # Adjust the crop boundaries
            x1 = round(x1 - width * 0.1)
            x2 = round(x2 + width * 0.1)
            y1 = round(y1)
            y2 = round(y2 + height * 0.3)

            # Ensure the crop is within bounds
            if not (0 <= x1 < cap.get(3) and 0 <= x2 <= cap.get(3) and 
                    0 <= y1 < cap.get(4) and 0 <= y2 <= cap.get(4)):
                print(f"Frame {frame_index}: Out of bounds, skipping.")
                frame_index += 1
                continue

            if x2 <= x1 or y2 <= y1:
                print(f"Frame {frame_index}: Invalid crop dimensions, skipping.")
                frame_index += 1
                continue

            # Crop, grayscale, resize
            croppedFrame = frame[y1:y2, x1:x2]
            grayFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)
            resizedFrame = cv2.resize(grayFrame, (fixed_width, fixed_height))

            # Write the frame multiple times if needed to match FPS
            if original_fps < target_fps:
                repeat_count = round(frame_interval)
                for _ in range(repeat_count):
                    out.write(resizedFrame)
                    output_frames += 1
            else:
                out.write(resizedFrame)
                output_frames += 1

        frame_index += 1

    # Ensure the final video has the correct duration
    expected_frames = int(target_fps * duration)
    while output_frames < expected_frames:
        out.write(resizedFrame)  # Duplicate last frame to match duration
        output_frames += 1

    # Release resources
    cap.release()
    out.release()

    print(f"Cropped video saved to {outputPath} with {target_fps} FPS")
