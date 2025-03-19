"""
This module load all seld feature into memory.
Reference code:  https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization
Note: there are two frame rates:
    1. feature frame rate: 80 frames/s
    2. label frame rate: 10 frames/s
"""
from rich.progress import Progress
import os
from typing import List
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Database:
    def __init__(self,
                 feat_label_dir : str = './DCASE2024_SELD_dataset/seld_feat_label/salsamic_dnorm_',
                 audio_format: str = 'foa', n_classes: int = 13, fs: int = 24000,
                 n_fft: int = 512, hop_len: int = 300, label_rate: float = 10, 
                 train_chunk_len_s: float = 5.0, train_chunk_hop_len_s: float = 2.5,
                 n_channels: int = 7, n_bins: int = 200,
                 training_folds = [3]):

        # Hardcoded, think of someway to fix soon
        for folder in os.listdir(feat_label_dir):
            if "norm" in folder:
                feat_folder = folder
            elif "adpit" in folder:
                label_folder = folder

        self.feature_root_dir = os.path.join(feat_label_dir, feat_folder)
        self.gt_meta_root_dir = os.path.join(feat_label_dir, label_folder)
        self.audio_format = audio_format
        self.n_classes = n_classes
        self.fs = fs
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.label_rate = label_rate
        self.train_chunk_len = self.second2frame(train_chunk_len_s)
        self.train_chunk_hop_len = self.second2frame(train_chunk_hop_len_s)
        self.training_folds = training_folds

        self.gt_chunk_len = int(train_chunk_len_s * 10) # label_rate * n_sec
        self.gt_chunk_hop_len = int(train_chunk_hop_len_s * 10)

        self.chunk_len = None
        self.chunk_hop_len = None
        self.feature_rate = self.fs / self.hop_len  # Frame rate per second
        self.label_upsample_ratio = int(self.feature_rate / self.label_rate)

        self.n_channels = n_channels
        self.n_bins = n_bins
        print("Feature shape : ({}, {}, {})".format(self.n_channels, self.train_chunk_len, self.n_bins))

    def second2frame(self, second: float = None) -> int:
        """
        Convert seconds to frame unit.
        """
        sample = int(second * self.fs)
        frame = int(round(sample/self.hop_len))
        return frame

    def get_segment_idxes(self, n_frames: int, downsample_ratio: int, pointer: int):
        # Get number of frame using segment rate
        assert n_frames % downsample_ratio == 0, 'n_features_frames is not divisible by downsample ratio'
        n_crop_frames = n_frames // downsample_ratio
        assert self.chunk_len // downsample_ratio <= n_crop_frames, 'Number of cropped frame is less than chunk len'

        idxes = np.arange(pointer,
                          pointer + n_crop_frames - self.chunk_len // downsample_ratio + 1,
                          self.chunk_hop_len // downsample_ratio).tolist()

        # Include the leftover of the cropped data
        if (n_crop_frames - self.chunk_len // downsample_ratio) % (self.chunk_hop_len // downsample_ratio) != 0:
            idxes.append(pointer + n_crop_frames - self.chunk_len // downsample_ratio)
        pointer += n_crop_frames

        return idxes, pointer


    def get_split(self):
        all_filenames = []
        for file in os.listdir(self.feature_root_dir):
            if file.endswith(".npy"):
                for training_fold in self.training_folds:
                    if "fold{}".format(training_fold) in file:
                        all_filenames.append(file)

        all_filenames = list(set(all_filenames))
        print("Total number of files : {}".format(len(all_filenames)))

        # Get chunk len and chunk hop len
        self.chunk_len = self.train_chunk_len
        self.chunk_hop_len = self.train_chunk_hop_len

        # Load and crop data
        features, labels, feature_chunk_idxes, gt_chunk_idxes, filename_list, test_batch_size = \
            self.load_chunk_data(split_filenames=all_filenames)
        # pack data
        db_data = {
            'features': features,
            'multi_accddoa_targets' : labels,
            'feature_chunk_idxes': feature_chunk_idxes,
            'gt_chunk_idxes': gt_chunk_idxes,
            'filename_list': filename_list,
            'test_batch_size': test_batch_size,
            'feature_chunk_len': self.chunk_len,
            'gt_chunk_len': self.chunk_len // self.label_upsample_ratio
        }

        print("Data loaded!")

        return db_data

    def load_chunk_data(self, split_filenames: List):
        feature_pointer = 0
        gt_pointer = 0
        features_list = []
        filename_list = []
        accdoa_target_list = []
        feature_idxes_list = []
        gt_idxes_list = []
        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]Loading files", total=len(split_filenames))

            for filename in split_filenames:
                # Load features -> n_channels x n_frames x n_features
                feature = np.load(os.path.join(self.feature_root_dir, filename))
                
                if feature.ndim == 2:
                    n_frames = feature.shape[0]
                    assert feature.shape[1] == int(self.n_channels * self.n_bins) , "Please check the feature space"
                    feature = feature.reshape(n_frames, self.n_channels, self.n_bins) # T, C, F
                    feature = feature.transpose((1,0,2)) # C, T, F

                # Load gt info from metadata
                accddoa = np.load(os.path.join(self.gt_meta_root_dir, filename))

                # We match the feature length with the number of ground truth labels that we have
                n_gt_frames = accddoa.shape[0]
                n_frames = n_gt_frames * 8
                if feature.shape[1] != n_frames : feature = feature[:, :n_frames]

                # Get sed segment indices
                feature_idxes, feature_pointer = self.get_segment_idxes(
                    n_frames=n_frames, downsample_ratio=1, pointer=feature_pointer)

                # Get gt segment indices
                gt_idxes, gt_pointer = self.get_segment_idxes(
                    n_frames=n_frames, downsample_ratio=self.label_upsample_ratio, pointer=gt_pointer) # Fixed at temporal downsample rate of 8x

                assert len(feature_idxes) == len(gt_idxes), 'nchunks for sed and gt are different'

                # Append data
                features_list.append(feature)
                filename_list.extend([filename] * len(feature_idxes))
                accdoa_target_list.append(accddoa)
                feature_idxes_list.append(feature_idxes)
                gt_idxes_list.append(gt_idxes)

                # Progress the progress bar
                progress.update(task, advance=1)

        if len(features_list) > 0:
            features = np.concatenate(features_list, axis=1)
            accddoa_targets = np.concatenate(accdoa_target_list, axis=0)
            feature_chunk_indexes = np.concatenate(feature_idxes_list, axis=0)
            gt_chunk_idxes = np.concatenate(gt_idxes_list, axis=0)
            test_batch_size = len(feature_idxes)  # to load all chunks of the same file
            return features, accddoa_targets, feature_chunk_indexes, gt_chunk_idxes, filename_list, test_batch_size
        else:
            return None, None, None, None, None, None

class seldDataset(Dataset):
    def __init__(self, db_data, transform=None):
        super().__init__()
        self.features = db_data['features']
        self.multi_accddoa_targets = db_data['multi_accddoa_targets']
        self.chunk_idxes = db_data['feature_chunk_idxes']
        self.gt_chunk_idxes = db_data['gt_chunk_idxes']
        self.filename_list = db_data['filename_list']
        self.chunk_len = db_data['feature_chunk_len']
        self.gt_chunk_len = db_data['gt_chunk_len']
        self.transform = transform  # transform that does not change label
        self.n_samples = len(self.chunk_idxes)
        
        print("seldDataset intiailized!\n\tNumber of batches : {}".format(self.n_samples))

    def __len__(self):
        """
        Total of training samples.
        """
        return self.n_samples

    def __getitem__(self, index):
        """
        Generate one sample of data
        """
        # Select sample
        chunk_idx = self.chunk_idxes[index]
        gt_chunk_idx = self.gt_chunk_idxes[index]

        # Load data and get label
        X = torch.tensor(self.features[:, chunk_idx: chunk_idx + self.chunk_len, :], dtype=torch.float32)
        target_labels = torch.tensor(self.multi_accddoa_targets[gt_chunk_idx:gt_chunk_idx + self.gt_chunk_len], dtype=torch.float32)

        if self.transform is not None:
            X = self.transform(X)

        return X, target_labels

class Iterated_Database:
    def __init__(self,
                 feat_label_dir: str = './DCASE2024_SELD_dataset/seld_feat_label/salsalite_no_dnorm_',
                 audio_format: str = 'foa', n_classes: int = 13, fs: int = 24000,
                 n_fft: int = 512, hop_len: int = 300, label_rate: float = 10,
                 train_chunk_len_s: float = 5.0, train_chunk_hop_len_s: float = 2.5,
                 n_channels: int = 7, n_bins: int = 191,
                 training_folds=[3]):

        # Determine feature and label folders
        for folder in os.listdir(feat_label_dir):
            if "norm" in folder:
                feat_folder = folder
            elif "adpit" in folder:
                label_folder = folder

        self.upper_feat_dir = feat_label_dir
        self.db_pickle_path = os.path.join(feat_label_dir, "db_data.pkl")
        self.feature_root_dir = os.path.join(feat_label_dir, feat_folder)
        self.gt_meta_root_dir = os.path.join(feat_label_dir, label_folder)
        self.audio_format = audio_format
        self.n_classes = n_classes
        self.fs = fs
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.label_rate = label_rate
        self.train_chunk_len = self.second2frame(train_chunk_len_s)
        self.train_chunk_hop_len = self.second2frame(train_chunk_hop_len_s)
        self.training_folds = training_folds

        self.gt_chunk_len = int(train_chunk_len_s * 10)  # label_rate * n_sec
        self.gt_chunk_hop_len = int(train_chunk_hop_len_s * 10)

        self.chunk_len = None
        self.chunk_hop_len = None
        self.feature_rate = self.fs / self.hop_len  # Frame rate per second
        self.label_upsample_ratio = int(self.feature_rate / self.label_rate)

        self.n_channels = n_channels
        self.n_bins = n_bins
        print("Feature shape : ({}, {}, {})".format(self.n_channels, self.train_chunk_len, self.n_bins))

    def second2frame(self, second: float = None) -> int:
        """
        Convert seconds to frame unit.
        """
        sample = int(second * self.fs)
        frame = int(round(sample / self.hop_len))
        return frame

    def get_segment_idxes(self, n_frames: int, downsample_ratio: int, pointer: int):
        # Get number of frame using segment rate
        assert n_frames % downsample_ratio == 0, 'n_features_frames is not divisible by downsample ratio'
        n_crop_frames = n_frames // downsample_ratio
        assert self.chunk_len // downsample_ratio <= n_crop_frames, 'Number of cropped frame is less than chunk len'

        idxes = np.arange(pointer,
                          pointer + n_crop_frames - self.chunk_len // downsample_ratio + 1,
                          self.chunk_hop_len // downsample_ratio).tolist()

        # Include the leftover of the cropped data
        if (n_crop_frames - self.chunk_len // downsample_ratio) % (self.chunk_hop_len // downsample_ratio) != 0:
            idxes.append(pointer + n_crop_frames - self.chunk_len // downsample_ratio)
        pointer += n_crop_frames

        return idxes, pointer

    def get_split(self):
        all_filenames = []
        for file in os.listdir(self.feature_root_dir):
            if file.endswith(".npy"):
                for training_fold in self.training_folds:
                    if "fold{}".format(training_fold) in file:
                        all_filenames.append(file)

        all_filenames = list(set(all_filenames))
        print("Total number of files : {}".format(len(all_filenames)))

        # Get chunk len and chunk hop len
        self.chunk_len = self.train_chunk_len
        self.chunk_hop_len = self.train_chunk_hop_len
        
        if os.path.exists(self.db_pickle_path):
            with open(self.db_pickle_path, "rb") as f:
                pickled_data = pickle.load(f)
                index_mappings = pickled_data['index_mapping']
            print("Index Mappings has been loaded from {}".format(self.db_pickle_path))
            
            # Pack data
            db_data = {
                'index_mapping': index_mappings,
                'feature_root_dir': self.feature_root_dir,
                'gt_meta_root_dir': self.gt_meta_root_dir,
                'feature_chunk_len': self.chunk_len,
                'gt_chunk_len': self.chunk_len // self.label_upsample_ratio,
                'label_upsample_ratio': self.label_upsample_ratio,
                'n_channels': self.n_channels,
                'n_bins': self.n_bins
            }

        else:
            
            print("db pickle does not exist. Creating index mappings now!")

            # Build index mappings without loading data
            index_mappings = self.build_index_mappings(all_filenames)

            # Pack data
            db_data = {
                'index_mapping': index_mappings,
                'feature_root_dir': self.feature_root_dir,
                'gt_meta_root_dir': self.gt_meta_root_dir,
                'feature_chunk_len': self.chunk_len,
                'gt_chunk_len': self.chunk_len // self.label_upsample_ratio,
                'label_upsample_ratio': self.label_upsample_ratio,
                'n_channels': self.n_channels,
                'n_bins': self.n_bins
            }

            print("Index mappings created!")
            
            with open(self.db_pickle_path, "wb") as fw:
                pickle.dump(db_data, fw)
            print("Saved db_data into {}".format(self.db_pickle_path))

        return db_data

    def build_index_mappings(self, split_filenames: List):
        index_mappings = []

        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]Processing files", total=len(split_filenames))

            for filename in split_filenames:
                # Load feature shape to get n_frames
                feature_path = os.path.join(self.feature_root_dir, filename)
                feature = np.load(feature_path)

                # Load label shape to get n_gt_frames
                gt_path = os.path.join(self.gt_meta_root_dir, filename)
                accdoa = np.load(gt_path)
                n_gt_frames = accdoa.shape[0]

                n_frames = n_gt_frames * self.label_upsample_ratio

                # Get segment indices
                feature_idxes, _ = self.get_segment_idxes(
                    n_frames=n_frames, downsample_ratio=1, pointer=0)

                gt_idxes, _ = self.get_segment_idxes(
                    n_frames=n_frames, downsample_ratio=self.label_upsample_ratio, pointer=0)

                assert len(feature_idxes) == len(gt_idxes), 'nchunks for feature and gt are different'

                for feature_idx, gt_idx in zip(feature_idxes, gt_idxes):
                    index_mappings.append({
                        'filename': filename,
                        'feature_chunk_idx': feature_idx,
                        'gt_chunk_idx': gt_idx
                    })

                if feature.ndim == 2:
                    feature = feature[:n_frames, :]
                    feature = feature.reshape(n_frames, self.n_channels, self.n_bins)  # T, C, F
                    feature = feature.transpose((1, 0, 2))  # C, T, F
                    np.save(feature_path,
                            feature)

                # Progress the progress bar
                progress.update(task, advance=1)

        return index_mappings


class Iterated_Dataset(Dataset):
    def __init__(self, db_data, transform=None):
        super().__init__()
        self.index_mapping = db_data['index_mapping']
        self.feature_root_dir = db_data['feature_root_dir']
        self.gt_meta_root_dir = db_data['gt_meta_root_dir']
        self.feature_chunk_len = db_data['feature_chunk_len']
        self.gt_chunk_len = db_data['gt_chunk_len']
        self.label_upsample_ratio = db_data['label_upsample_ratio']
        self.n_channels = db_data['n_channels']
        self.n_bins = db_data['n_bins']
        self.transform = transform  # transform that does not change label
        self.n_samples = len(self.index_mapping)

        print("seldDataset initialized!\n\tNumber of samples : {}".format(self.n_samples))

    def __len__(self):
        """
        Total number of training samples.
        """
        return self.n_samples

    def __getitem__(self, index):
        """
        Generate one sample of data
        """
        # Retrieve mapping information
        mapping = self.index_mapping[index]
        filename = mapping['filename']
        feature_chunk_idx = mapping['feature_chunk_idx']
        gt_chunk_idx = mapping['gt_chunk_idx']

        # Load feature chunk
        feature_path = os.path.join(self.feature_root_dir, filename)
        feature = np.load(feature_path, mmap_mode='r')

        # Extract chunk
        X = feature[:, feature_chunk_idx: feature_chunk_idx + self.feature_chunk_len, :]

        # Load label chunk
        gt_path = os.path.join(self.gt_meta_root_dir, filename)
        accdoa = np.load(gt_path, mmap_mode='r')
        target_labels = accdoa[gt_chunk_idx: gt_chunk_idx + self.gt_chunk_len]

        if self.transform is not None:
            X = self.transform(X)

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        target_labels = torch.tensor(target_labels, dtype=torch.float32)

        return X, target_labels


if __name__ == '__main__':

    db = Iterated_Database()
    db_data = db.get_split()
    dataset = Iterated_Dataset(db_data)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True, shuffle=True)
    
    print('Number of batches: {}'.format(len(dataloader)))
    
    for train_iter, (X, targets, filenames) in enumerate(dataloader):
        print("Iteration {}/{}".format(train_iter+1, len(dataloader)), end='\r')
        assert X.shape == (32, 7, 400, 191), "Something wrong at {} --> {}".format(train_iter, X.shape)
        assert targets.shape == (32, 50, 6, 5, 13), "Something wrong at {} --> {}".format(train_iter, targets.shape)
        # if len(set(filenames)) != 1:
        #     print("{} : {}".format(train_iter + 1, set(filenames)))
        if train_iter + 1 == len(dataloader):
            print('X: dtype: {} - shape: {}'.format(X.dtype, X.shape))
            print('Multi-ACCDDOA: dtype: {} - shape: {}'.format(targets.dtype, targets.shape))
            print(set(filenames), len(set(filenames)))