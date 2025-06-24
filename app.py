import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random

class UCF101Dataset(Dataset):
    def __init__(self, root_dir, classes, transform=None, frames_per_clip=32, mode='train'):
        self.root_dir = root_dir
        self.classes = classes
        self.transform = transform
        self.frames_per_clip = frames_per_clip  # Updated default to 32
        self.mode = mode  # 'train' or 'val/test'

        self.video_paths = []
        self.labels = [] 

        for class_idx, class_name in enumerate(classes):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue
            for video_file in os.listdir(class_folder):
                if video_file.endswith(".avi"):
                    video_path = os.path.join(class_folder, video_file)
                    self.video_paths.append(video_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Shape: (frames_per_clip, C, H, W) -> (C, D, H, W)
        frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames == 0 or not cap.isOpened():
            # Return black frames if video can't be read
            return [np.zeros((112, 112, 3), dtype=np.uint8)] * self.frames_per_clip

        # Choose start frame based on mode
        if total_frames > self.frames_per_clip:
            if self.mode == 'train':
                start = random.randint(0, total_frames - self.frames_per_clip)
            else:  # validation or test
                start = (total_frames - self.frames_per_clip) // 2
        else:
            start = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        count = 0
        while count < self.frames_per_clip and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            count += 1

        cap.release()

        # If fewer frames, pad with the last frame or black
        if len(frames) == 0:
            frames = [np.zeros((112, 112, 3), dtype=np.uint8)] * self.frames_per_clip
        while len(frames) < self.frames_per_clip:
            frames.append(frames[-1])

        return frames
