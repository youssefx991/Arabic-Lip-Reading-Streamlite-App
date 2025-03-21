import cv2
from pymediainfo import MediaInfo

def get_video_rotation_mediainfo(video_path):
    media_info = MediaInfo.parse(video_path)
    for track in media_info.tracks:
        if track.track_type == "Video" and track.rotation:
            return int(float(track.rotation))
    return 0  # No rotation detected

def rotate(videoPath, outputPath="final_video.mp4", rotation=0):
    print("in rotate from Rotate")

    # Load the video
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error opening video file:", videoPath)
        return
    print("video path for rotation:", videoPath)
    print(f"Rotation: {rotation} degrees")

    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Adjust width and height for rotation
    if rotation in [90, 270]:
        width, height = height, width

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputPath, fourcc, original_fps, (width, height))

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate the frame
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Write the frame to the output video
        out.write(frame)
        frame_index += 1

    # Release everything if job is finished
    cap.release()
    out.release()
    print("Rotation complete. Output saved to:", outputPath)