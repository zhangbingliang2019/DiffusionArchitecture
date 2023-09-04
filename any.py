# import os
# file = ["1066674784.mp4",
# "1066674790.mp4",
# "1066674802.mp4",
# "1066674877.mp4",
# "1066674883.mp4",
# "1066674916.mp4",
# "1066674928.mp4",
# "1066674931.mp4",
# "1066674934.mp4",
# "1066674937.mp4",
# "1066674940.mp4",
# "1066674943.mp4",
# "1066674946.mp4",
# "1066674949.mp4",
# "1066674952.mp4",
# "1066674958.mp4"]
#
# command = ("scp -r -P 61201 bingliang@rp-a100-"
#            "80gb-8x-hba001.cloud.together.ai:"
#            "/home/bingliang/data/WebVid2.5M/videos/000001_000050/{} videos")
#
# for p in file:
#     print(p)
#     os.system(command.format(p))

# filename = 'videos/1066674784.mp4'
# import skvideo.io
# videodata = skvideo.io.vread(filename)
# print(videodata.shape)

import pandas as pd
import tqdm
import os
from torchvision.io import read_video
import json
import numpy as np

# path = '/home/bingliang/data/webvid/results_2M_train.csv'
# meta = pd.read_csv(path)
# info = []
# frame_size = []
#
# # filter out extra data
# for i in tqdm.trange(meta.shape[0]):
#     feature = meta.iloc[i]
#     # print(feature)
#     # print(np.asarray(feature))
#     if isinstance(feature.iloc[3], str) and int(feature.iloc[3].split('_')[-1]) <= 10000:
#         video_id, caption, page_dir = str(feature.iloc[0]),str(feature.iloc[1]), str(feature.iloc[3])
#         video_path = '/home/bingliang/data/WebVid2.5M/videos/{}/{}.mp4'.format(page_dir, video_id)
#         if os.path.exists(video_path):
#             info.append({'video_id': video_id, 'caption': caption, 'page_dir': page_dir})
#             # video_data = read_video(video_path, pts_unit='sec', output_format='TCHW')[0]
#             # frame_size.append(video_data.size(0))

import json
import ffmpeg
import pprint
import multiprocessing as mp

new_info = []


def test(start, end):
    print('in: ', start, end)
    F, H, W = [], [], []
    info = json.load(open('/home/bingliang/data/WebVid2.5M/subset_info.json', 'r'))
    end = min(len(info), end)
    for i in tqdm.trange(start, end):
        video_id, caption, page_dir = info[i].values()
        video_path = '/home/bingliang/data/WebVid2.5M/videos/{}/{}.mp4'.format(page_dir, video_id)
        try:
            meta = ffmpeg.probe(video_path)['streams']
            if 'nb_frames' in meta[0] and 'height' in meta[0] and 'width' in meta[0]:
                frame, height, width = int(meta[0]['nb_frames']), meta[0]['height'], meta[0]['width']
                if frame >= 64 and height == 336 and width == 596:
                    new_info.append(info[i])
        except:
            continue


test(0, 554000)
print(len(new_info))
json.dump(new_info, open('/home/bingliang/data/WebVid2.5M/subset_new_info.json', 'w'), indent=2)
