from torch.utils.data import Dataset
import torch
import json
from decord import VideoLoader, VideoReader
from decord import cpu, gpu
import numpy as np



# Video Data Augmentation
# def center_crop(t_video, image_size):
#     # t_video: (F, C, H, W)
#     scale = image_size / min(*t_video.shape[2:])
#     resized_t_video = F.interpolate(t_video, scale_factor=scale, mode="bilinear")
#
#     crop_x = (resized_t_video.size(2) - image_size) // 2
#     crop_y = (resized_t_video.size(3) - image_size) // 2
#     return resized_t_video[:, :, crop_x:crop_x + image_size, crop_y:crop_y + image_size]
#
#
# def frame_sampling(t_video, frame_size):
#     # t_video: (F, C, H, W)
#     sampled_frame = (torch.linspace(0, t_video.size(0), frame_size + 1)[:frame_size] + 0.5).to(torch.int)
#     return t_video[sampled_frame]
#
#
# class WebVid(Dataset):
#     def __init__(self, video_root, json_path, image_size=256, frame_size=32, overfitting_test=False):
#         super().__init__()
#         # ['videoid', 'name', 'page_idx', 'page_dir', 'duration', 'contentUrl']
#         self.info = json.load(open(json_path, 'r'))  # list of dict: {video_id, caption, page_dir},
#         self.video_root = video_root
#         self.frame_size = frame_size
#         self.image_size = image_size
#         self.test = overfitting_test
#
#         # # filter out extra data
#         # for i in tqdm.trange(meta.shape[0]):
#         #     feature = meta.iloc[i]
#         #     # print(feature)
#         #     # print(np.asarray(feature))
#         #     if isinstance(feature.iloc[3], str) and int(feature.iloc[3].split('_')[-1]) <= group_limit:
#         #         self.info.append((feature.iloc[0], feature.iloc[1], feature.iloc[3]))
#
#     def __getitem__(self, item):
#         video_id, caption, page_dir = self.info[item].values()
#         video_path = self.video_root + '/{}/{}.mp4'.format(page_dir, video_id)
#         video_data = read_video(video_path, pts_unit='sec', output_format='TCHW')[0]  # (L, C, H, W)
#         processed_video_data = center_crop(frame_sampling(video_data / 255.0, self.frame_size), self.image_size)
#         # normalize
#         normalized_video_data = processed_video_data * 2 - 1
#         return normalized_video_data, caption
#
#     def __len__(self):
#         if self.test:
#             return 4
#         return len(self.info)

class WebVidFast(Dataset):
    """
        all videos are of resolution (336, 596)
    """
    def __init__(self, video_root, json_path, image_size=336, frame_size=16,
                 overfitting_test=False, return_caption=False):
        super().__init__()
        self.info = json.load(open(json_path, 'r'))  # list of dict: {video_id, caption, page_dir},
        self.video_root = video_root
        self.frame_size = frame_size
        self.image_size = image_size
        self.return_caption = return_caption
        if overfitting_test:
            self.info = [self.info[i] for i in range(16)]
        self.width = int(self.image_size / 336.0 * 596.0)
        self.start = (self.width - self.image_size) // 2

    def load_video(self, path):
        vr = VideoReader(path, ctx=cpu(0))
        frames_num = len(vr)
        sampled_frame = np.linspace(0, frames_num, self.frame_size, endpoint=False).astype(int)
        frames = vr.get_batch(sampled_frame)
        return frames

    def load_video_by_loader(self, path):

        vl = VideoLoader([path], ctx=[cpu(0)], shape=(self.frame_size, self.image_size, self.width, 3),
                         interval=1, skip=5, shuffle=0)
        return self.center_crop(vl.next()[0].asnumpy())

    def center_crop(self, video):
        # (B, H, W, C)
        return video[:, :, self.start:self.start+self.image_size]


    def __getitem__(self, item):
        """
            (-1, 1) range images (F, C, H, W)
        """
        video_id, caption, page_dir = self.info[item].values()
        video_path = self.video_root + '/{}/{}.mp4'.format(page_dir, video_id)

        frames = self.load_video_by_loader(video_path)
        # normalize
        if not self.return_caption:
            return torch.from_numpy(frames).permute(0, 3, 1, 2) / 255 * 2 - 1
        return torch.from_numpy(frames).permute(0, 3, 1, 2) / 255 * 2 - 1, caption

    def __len__(self):
        return len(self.info)



if __name__ == '__main__':
    from video_dataset import WebVid

    dataset = WebVid("/home/bingliang/data/WebVid2.5M/videos",
                     "/home/bingliang/data/WebVid2.5M/subset_new_info.json")
    print(len(dataset))
