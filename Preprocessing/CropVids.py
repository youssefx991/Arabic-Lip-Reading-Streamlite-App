import cv2
import os

def cropVideo(point1, point2, videoPath, outputPath="cropped_video.mp4"):
    """
    Crop video based on top left point (point1) and bottom right point (point2) coordinates.
    The cropped video is converted to grayscale, resized to a fixed resolution (65x40), and then saved.

    Args:
        point1 ((int,int)): Top-left point coordinates.
        point2 ((int,int)): Bottom-right point coordinates.
        videoPath (str): Path to the input video.
        outputPath (str): Path to save the cropped video.
    """
    
    x1, y1 = point1[0], point1[1]  # Top-left corner
    x2, y2 = point2[0], point2[1]  # Bottom-right corner

    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # Adjust the crop boundaries slightly (optional)
    x1 = round(x1 - width * 0.15)
    x2 = round(x2 + width * 0.15)
    y1 = round(y1 - height * 0.20)
    y2 = round(y2 + height * 0.20)

    # Load the video
    cap = cv2.VideoCapture(videoPath)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if the new coordinates are within the video frame boundaries
    if not (0 <= x1 < frame_width and 0 <= x2 <= frame_width and 0 <= y1 < frame_height and 0 <= y2 <= frame_height):
        print("Out of image indices!##############################################")
        print("Out of image indices!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        return
    elif x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid cropping dimensions: Ensure x2 > x1 and y2 > y1.")

    # Define the fixed size for the output video (width=65, height=40)
    fixed_width, fixed_height = 65, 40

    # Define the output video writer.
    # Setting isColor=False because we are writing grayscale frames.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(outputPath, fourcc, fps, (fixed_width, fixed_height), isColor=False)

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame using the defined coordinates
        croppedFrame = frame[y1:y2, x1:x2]

        # Convert the cropped frame to grayscale
        grayFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2GRAY)

        # Resize the grayscale frame to fixed dimensions: width=65 and height=40
        resizedFrame = cv2.resize(grayFrame, (fixed_width, fixed_height))

        # Write the resized grayscale frame to the output video
        out.write(resizedFrame)

    # Release resources
    cap.release()
    out.release()

    # print(f"Cropped video saved to {outputPath}")
