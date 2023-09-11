import cv2
from torch.utils.data import Dataset
import torch
import json
from PIL import Image
import numpy as np


# Video Data Augmentation
def center_crop(lists, image_size):
    # n_video: (F, H, W, C)
    new_list = []
    H, W = lists[0].shape[0:2]
    scale = image_size / min(H, W)
    rH, rW = int(scale * H), int(scale * W)
    crop_x = (rH - image_size) // 2
    crop_y = (rW - image_size) // 2
    for image in lists:
        resize_i = np.asarray(Image.fromarray(image).resize(size=(rW, rH), resample=Image.Resampling.LANCZOS))
        new_list.append(resize_i[crop_x:crop_x + image_size, crop_y:crop_y + image_size])
    return np.stack(new_list)


def frame_sampling(t_video, frame_size):
    # t_video: (F, C, H, W)
    sampled_frame = (torch.linspace(0, t_video.size(0), frame_size + 1)[:frame_size] + 0.5).to(torch.int)
    return t_video[sampled_frame]


class WebVid(Dataset):
    def __init__(self, video_root, json_path, image_size=256, frame_size=32,
                 overfitting_test=False, return_caption=True):
        super().__init__()
        # ['videoid', 'name', 'page_idx', 'page_dir', 'duration', 'contentUrl']
        self.info = json.load(open(json_path, 'r'))  # list of dict: {video_id, caption, page_dir},
        self.video_root = video_root
        self.frame_size = frame_size
        self.image_size = image_size
        self.test = overfitting_test
        self.return_caption = return_caption

        # # filter out extra data
        # for i in tqdm.trange(meta.shape[0]):
        #     feature = meta.iloc[i]
        #     # print(feature)
        #     # print(np.asarray(feature))
        #     if isinstance(feature.iloc[3], str) and int(feature.iloc[3].split('_')[-1]) <= group_limit:
        #         self.info.append((feature.iloc[0], feature.iloc[1], feature.iloc[3]))

    def load_video(self, path):
        vc = cv2.VideoCapture(path)
        frames_num = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frame = (torch.linspace(0, frames_num, self.frame_size + 1)[:self.frame_size] + 0.5).to(torch.int)
        frames = []
        if vc.isOpened():
            ret, frame = vc.read()
        count = 0
        idx = 0
        while ret:
            while idx < len(sampled_frame) and sampled_frame[idx] == count:
                frames.append(frame)
                idx += 1
            ret, frame = vc.read()
            count += 1
        vc.release()
        return frames

    def __getitem__(self, item):
        """
            (0, 1) range images (C, F, H, W)
        """
        video_id, caption, page_dir = self.info[item].values()
        video_path = self.video_root + '/{}/{}.mp4'.format(page_dir, video_id)
        np_videos = center_crop(self.load_video(video_path), self.image_size)

        video = torch.from_numpy(np_videos).permute(0, 3, 1, 2).float() / 255 * 2 - 1
        if self.return_caption:
            return video, caption
        else:
            return video

    def __len__(self):
        if self.test:
            return 4
        return len(self.info)


def extract_frame(video_path: str):
    vc = cv2.VideoCapture(video_path)
    frames_num = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frames_num)
    if vc.isOpened():
        ret, frame = vc.read()
    count = 1
    while ret:
        count += 1
        ret, frame = vc.read()
    vc.release()
    print(count)


if __name__ == '__main__':
    import tqdm
    import numpy as np

    dataset = WebVid("/home/bingliang/data/WebVid2.5M/videos",
                     "/home/bingliang/data/WebVid2.5M/subset_new_info.json")

    frame = []
    for i in tqdm.trange(30):
        frame.append(dataset[i])
    print(frame[0].shape)
