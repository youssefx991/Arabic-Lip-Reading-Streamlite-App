import os
import tkinter as tk
from tkinter import filedialog

def dirExists(dir_path):
    os.makedirs(dir_path, exist_ok=True) # Create directory if it does not exist.

def getFilePath():
    print('in getFilePath()')
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select File")  # Open file dialog
    root.destroy() # Destroy the root window to prevent it from staying.
    return file_path

def getDirPath():
    print('in getDirPath()')
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    folder_paths = filedialog.askdirectory(
        title="Select Folders",  # Dialog title
        mustexist=True  # Ensure selected folders exist
    )
    root.destroy() # Destroy the root window to prevent it from staying.
    return folder_paths

def getVidName(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print('video_name: ',video_name)
    return video_name

