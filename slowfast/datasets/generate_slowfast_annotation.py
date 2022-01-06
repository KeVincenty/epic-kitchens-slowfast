import pandas as pd
from datetime import timedelta
import time
import numpy as np
import csv
import subprocess
import os

np.random.seed(0)

video_info_path = '/home/yinyuan/workspace/epic-kitchens-100-annotations/EPIC_100_video_info.csv'
train_file_path = '/home/yinyuan/workspace/epic-kitchens-100-annotations/EPIC_100_train.pkl'
val_file_path = '/home/yinyuan/workspace/epic-kitchens-100-annotations/EPIC_100_validation.pkl'
out_train_path = '/home/yinyuan/workspace/epic-kitchens-slowfast/annotations/epic-kitchens/EPIC_100_train.pkl'
out_val_path = '/home/yinyuan/workspace/epic-kitchens-slowfast/annotations/epic-kitchens/EPIC_100_validation.pkl'

video_info = {}
video_length = {}
i = 0
with open('/home/yinyuan/workspace/epic-kitchens-100-annotations/EPIC_100_video_info.csv','r') as f:
    data = csv.reader(f)
    for line in data:
        i+=1
        if i > 1:
            video_info[line[0]] = {'frames':int(np.ceil(float(line[1])*float(line[2]))),'fps':float(line[2])}

for k,_ in video_info.items():
    video_prefix = k.split('_')[0]
    video_path = os.path.join('/work/share/EPIC_KITCHENS_100/EPIC-KITCHENS/', video_prefix, 'rgb_frames', k)
    frames = subprocess.check_output('ls -lh ' + video_path + ' | wc -l', shell=True)
    frames = int(frames)
    assert frames > 0
    video_length[k] = frames


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec

def generate(input_path, output_path):
    single_timestamp_list = []
    frames_to_end_list = []
    data = pd.read_pickle(input_path)

    for row in data.iterrows():
        index = row[0]
        series = row[1]
        if len(series['video_id'].split('_')[1]) == 3:
            fps = 50
        else:
            fps = 60
        start_frame = int(round(timestamp_to_sec(series['start_timestamp']) * fps))
        end_frame = int(round(timestamp_to_sec(series['stop_timestamp']) * fps))
        timestamp_frame = np.random.randint(start_frame, end_frame)
        single_timestamp_list.append(timestamp_frame)
        frames_of_video = video_length[series['video_id']]
        frames_to_end = frames_of_video - timestamp_frame -1
        frames_to_end_list.append(frames_to_end)
    
    data['single_timestamp'] = single_timestamp_list
    data['frames_to_end'] = frames_to_end_list
    data.to_pickle(output_path)

generate(train_file_path, out_train_path)
generate(val_file_path, out_val_path)
