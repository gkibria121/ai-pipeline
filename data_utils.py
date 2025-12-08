import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from feature_extraction import FeatureExtractor

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, feature_type=0):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.feature_type = feature_type
        self.extractor = FeatureExtractor()
        self.sample_rate = 16000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        filepath = self.base_dir / f"{utt_id}.flac"
        
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            audio = np.zeros(64600)
        
        feature = self.extractor.extract_feature(audio, self.feature_type)
        feature = torch.FloatTensor(feature)
        label = self.labels[utt_id]
        return feature, label


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, feature_type=0):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.feature_type = feature_type
        self.extractor = FeatureExtractor()
        self.sample_rate = 16000

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        filepath = self.base_dir / f"{utt_id}.flac"
        
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            audio = np.zeros(64600)
        
        feature = self.extractor.extract_feature(audio, self.feature_type)
        feature = torch.FloatTensor(feature)
        return feature, utt_id
