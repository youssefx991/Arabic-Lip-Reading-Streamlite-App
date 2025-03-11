from moviepy.editor import VideoFileClip, AudioClip

def standardizeFPS(video: VideoFileClip, targetFPS: int = 25) -> VideoFileClip:
    if video.fps == targetFPS:
        print("No need to change the fps")
        return video

    # Change the FPS
    video_with_new_fps = video.set_fps(targetFPS)  # Set to your desired FPS, e.g., 30 FPS

    # add silent_audio audio
    silent_audio = AudioClip(make_frame=lambda t: 0)  # Creates a silent audio clip
    video_with_new_fps = video_with_new_fps.set_audio(silent_audio)

    return video_with_new_fps
