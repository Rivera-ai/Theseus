import csv
import os
import random
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import json
from VideoData import ToTensorVideo, UCFCenterCropVideo

""" Igual aqui dejo el dataset tanto para CSV como para JSON """

def GetTransformsVideo(resolution=256):
    videotransf = transforms.Compose([ToTensorVideo(), UCFCenterCropVideo(resolution),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5],
                             inplace=True), ])
    return videotransf

def GetRowFilter(filter_type: int):
    def func(row):
        return True

    def func1(row):
        data_category = row[3]
        return data_category == "NSFW"
    
    if filter_type == 1:
        return func1
    else:
        return func


class DatasetFromJSON(torch.utils.data.Dataset):
    def __init__(self,
                 json_path,
                 num_frames=16,
                 frame_interval=1,
                 transform=None,
                 root=None):
        self.samples = []
        self.json_path = json_path
        self.root = root
        
        if not os.path.exists(json_path) and root is not None:
            self.json_path = os.path.join(self.root, json_path)

        # Load JSON data
        with open(self.json_path) as f:
            data = json.load(f)
            self.samples = data

        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.num_real_frames = 1 + (num_frames - 1) * frame_interval

    def getitem(self, index):
        sample = self.samples[index]
        video_path = os.path.join(self.root, sample['video_path']) if self.root else sample['video_path']
        text = sample['text']

        # Read video frames
        vframes, _, _ = torchvision.io.read_video(
            filename=video_path, 
            pts_unit="sec", 
            output_format="TCHW"
        )

        # Sample frames
        total_frames = len(vframes)
        start_frame_ind = 0
        end_frame_ind = start_frame_ind + self.num_real_frames
        frame_indices = np.arange(start_frame_ind,
                                end_frame_ind,
                                step=self.frame_interval,
                                dtype=int)
        video = vframes[frame_indices]

        if self.transform:
            video = self.transform(video)  # T C H W
        
        video = video.permute(1, 0, 2, 3)  # C T H W, channel first convention

        return {
            "video": video,
            "text": text,
        }

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(f"Error loading sample {index}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


class PreprocessedDatasetFromJSON(torch.utils.data.Dataset):
    def __init__(self,
                 json_path,
                 num_frames=None,
                 root=None,
                 preprocessed_dir=None):
        self.samples = []
        self.json_path = json_path
        self.num_frames = num_frames
        self.root = root
        
        if not os.path.exists(json_path) and root is not None:
            self.json_path = os.path.join(self.root, json_path)

        self.preprocessed_dir = preprocessed_dir
        if not os.path.exists(preprocessed_dir) and root is not None:
            self.preprocessed_dir = os.path.join(self.root, preprocessed_dir)

        # Load JSON data
        with open(self.json_path) as f:
            data = json.load(f)
            self.samples = data

    def getitem(self, index):
        sample = self.samples[index]
        video_id = sample['video_id']

        preprocessed_data_path = os.path.join(self.preprocessed_dir, f"{video_id}.pt")
        data = torch.load(preprocessed_data_path)

        if self.num_frames is not None:
            data['x'] = data['x'][:, :self.num_frames]

        return data

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(f"Error loading sample {index}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)

class DatasetFromCSV(torch.utils.data.Dataset):
    def __init__(self,
                 csv_path,
                 num_frames=16,
                 frame_interval=1,
                 transform=None,
                 root=None,
                 data_filter=None):
        self.samples = []
        self.csv_path = csv_path
        self.root = root
        
        #print(f"Initializing Dataset with:")
        #print(f"CSV Path: {csv_path}")
        #print(f"Root: {root}")
        #print(f"Frames: {num_frames}")
        #print(f"Interval: {frame_interval}")
        
        if not os.path.exists(csv_path) and root is not None:
            self.csv_path = os.path.join(self.root, csv_path)
            #print(f"Updated CSV path: {self.csv_path}")

        row_filter = GetRowFilter(data_filter)
        #print("Reading CSV file...")
        
        try:
            with open(self.csv_path) as f:
                reader = csv.reader(f)
                headers = next(reader)
                #print(f"CSV Headers: {headers}")
                
                for i, row in enumerate(reader):
                    #print(f"Processing row {i}: {row}")
                    if row_filter(row):
                        self.samples.append(row)
                        #print(f"Row {i} accepted")
                    else:
                        print(f"Row {i} filtered out")
                        
                print(f"Total samples accepted: {len(self.samples)}")
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")

        self.transform = transform
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.num_real_frames = 1 + (num_frames - 1) * frame_interval
        #print(f"Dataset initialized with {len(self.samples)} samples")

    def getitem(self, index):
        t0 = time.time()
        try:
            video_id, url, duration, page_dir, text = self.samples[index]
            #print(f"\nAccessing item {index}:")
            #print(f"Video ID: {video_id}")
            #print(f"Page Dir: {page_dir}")
            
            if self.root:
                path = os.path.join(self.root, f"{video_id}")
                #print(f"Full video path: {path}")

            short_text = ' '.join(video_id.split('-')[1:-1])
            video_category = page_dir.split("/")[-1]
            #print(f"Category: {video_category}")

            #print("Reading video file...")
            vframes, aframes, info = torchvision.io.read_video(
                filename=path, pts_unit="sec", output_format="TCHW")
            #print(f"Video loaded, total frames: {len(vframes)}")

            total_frames = len(vframes)
            start_frame_ind = 0
            end_frame_ind = start_frame_ind + self.num_real_frames
            frame_indice = np.arange(start_frame_ind, end_frame_ind, 
                                   step=self.frame_interval, dtype=int)
            video = vframes[frame_indice]
            #print(f"Selected {len(frame_indice)} frames")

            if self.transform:
                video = self.transform(video)  # T C H W
                #print("Applied transforms")

            video = video.permute(1, 0, 2, 3)  # C T H W
            #print(f"Final video tensor shape: {video.shape}")
            #print(f"Time taken: {time.time() - t0:.2f}s")

            return {
                "video": video,
                "text": text,
                "short_text": short_text,
                "category": video_category,
                "video_id": video_id,
            }
        except Exception as e:
            print(f"Error in getitem for index {index}: {str(e)}")
            print(f"Sample data: {self.samples[index]}")
            raise e

    def __getitem__(self, index):
        for attempt in range(5):
            try:
                #print(f"\nAttempt {attempt + 1} for index {index}")
                return self.getitem(index)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {str(e)}")
                index = np.random.randint(len(self))
                print(f"Trying new random index: {index}")
        raise RuntimeError("Too many bad data attempts")

    def __len__(self):
        return len(self.samples)


class PreprocessedDatasetFromCSV(torch.utils.data.Dataset):

    def __init__(self,
                 csv_path,
                 num_frames=None,
                 root=None,
                 preprocessed_dir=None,
                 data_filter=1):
        # import pandas
        self.samples = []
        self.csv_path = csv_path
        self.num_frames = num_frames
        self.root = root
        if not os.path.exists(csv_path) and root is not None:
            self.csv_path = os.path.join(self.root, csv_path)

        self.preprocessed_dir = preprocessed_dir
        if not os.path.exists(preprocessed_dir) and root is not None:
            self.preprocessed_dir = os.path.join(self.root, preprocessed_dir)

        row_filter = GetRowFilter(data_filter)
        with open(self.csv_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row_filter(row):
                    self.samples.append(row)
            print(len(self.samples))

    def getitem(self, index):
        t0 = time.time()
        video_id, url, duration, page_dir, text = self.samples[index]

        preprocessed_data_path = os.path.join(self.preprocessed_dir, f"{video_id}.pt")
        data = torch.load(preprocessed_data_path)
        # x = data['x'] # C T H W, channel first convention
        # y = data['y']
        # mask = data['mask']
        if self.num_frames is not None:
            data['x'] = data['x'][:, :self.num_frames]

        return data

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)