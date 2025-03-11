from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

def standardizeDuration(video: VideoFileClip, targetDuration: int = 1) -> VideoFileClip:
    """
    Change the video length while keeping the same fps => change the number of frames.
    """

    if video is None:
        print("Error: video is None.")
        return None

    # Specify target duration in seconds
    current_duration = video.duration
    if current_duration == targetDuration:
        print("No need to change the duration")
        return video
        
    stretch_factor = targetDuration / current_duration

    # Stretch the video (adjust speed to match target duration)
    video_stretched = video.fx(vfx.speedx, factor=1/stretch_factor)

    # silent_audio = AudioClip(make_frame=lambda t: 0)  # Creates a silent audio clip
    # video_stretched = video_stretched.set_audio(silent_audio)

    return video_stretched
