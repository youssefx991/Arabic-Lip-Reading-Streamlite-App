from Pre_main02 import preprocess
import os

video_path = r'H:\FCIS\GP\Arabic-Lip-Reading\Preprocessing\test_video0000.mp4'

if os.path.exists(video_path):
    print('File found')
    preprocess(video_path=video_path, word='test', user='test_user')
else:
    print('File not found')