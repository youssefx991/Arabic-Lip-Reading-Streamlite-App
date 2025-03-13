import cv2
import os
import streamlit as st

def cropVideo(topLeftPoints, bottomRightPoints, videoPath, outputPath="cropped_video.mp4"):
    """
    Crop each frame of the video using per-frame top-left and bottom-right coordinates.
    The cropped region of each frame is converted to grayscale, resized to a fixed 
    resolution (65x40), and written to an output video file.

    Args:
        topLeftPoints (list of (int,int)): List of top-left corner coordinates per frame.
        bottomRightPoints (list of (int,int)): List of bottom-right corner coordinates per frame.
        videoPath (str): Path to the input video.
        outputPath (str): Path to save the cropped video.
    """
    # Load the video
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        st.info("Error opening video file:", videoPath)
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the fixed output size for the cropped video
    fixed_width, fixed_height = 160,150

    # Define the output video writer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(outputPath, fourcc, fps, (fixed_width, fixed_height), isColor=False)

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the corresponding points for this frame.
        # If there are fewer points than frames, use the last available points.
        if frame_index < len(topLeftPoints) and frame_index < len(bottomRightPoints):
            TL = topLeftPoints[frame_index]
            BR = bottomRightPoints[frame_index]
        else:
            TL = topLeftPoints[-1]
            BR = bottomRightPoints[-1]

        # Extract coordinates
        x1, y1 = TL[0], TL[1]
        x2, y2 = BR[0], BR[1]

        # Calculate width and height of the region
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Adjust the crop boundaries (optional padding)
        x1 = round(x1 - width * 0.1)
        x2 = round(x2 + width * 0.1)
        y1 = round(y1)
        y2 = round(y2 + height * 0.3)

        # Check if the new coordinates are within the video frame boundaries
        if not (0 <= x1 < frame_width and 0 <= x2 <= frame_width and 
                0 <= y1 < frame_height and 0 <= y2 <= frame_height):
            # st.info(f"Frame {frame_index}: Out of image indices! Skipping this frame.")
            frame_index += 1
            continue

        if x2 <= x1 or y2 <= y1:
            # st.info(f"Frame {frame_index}: Invalid cropping dimensions. Skipping this frame.")
            frame_index += 1
            continue

        # Crop the frame using the defined coordinates
        croppedFrame = frame[y1:y2, x1:x2]

        # Convert the cropped frame to grayscale
        grayFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale frame to fixed dimensions: width=65 and height=40
        resizedFrame = cv2.resize(grayFrame, (fixed_width, fixed_height))

        # Write the resized grayscale frame to the output video
        out.write(resizedFrame)

        frame_index += 1

    # Release resources
    cap.release()
    out.release()

    _ = ''' st.info(f"Cropped video saved to {outputPath}") '''