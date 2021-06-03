import os
import numpy as np
import shutil
import subprocess
import glob
from tqdm import tqdm

video_path = r'./data/TrainValVideo/'
# video_path = r'./data/TestVideo/'


def extract_frames(video, dst):
    with open(os.devnull, "w") as ffmpeg_log:
        if os.path.exists(dst):
            print(" cleanup: " + dst + "/")
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ["ffmpeg",
                                   # (optional) overwrite output file if it exists
                                   '-y',
                                   '-i', video,  # input file
                                   '-vf', "scale=400:300",  # input file
                                   '-qscale:v', "2",  # quality for JPEG
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


video_list = glob.glob(os.path.join(video_path, '*.mp4'))

for video in tqdm(video_list):
    video_id = video.split("/")[-1].split(".")[0]
    dst = os.path.join('./video2images', video_id)
    # print('extract images from %s...' % video_id)
    extract_frames(video, dst)

print('extract images from videos successfully!')

