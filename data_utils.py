import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from feature_extraction import FeatureExtractor
from pathlib import Path

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
    def __init__(self, list_IDs, labels, base_dir, feature_type=0, cache_dir=None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = Path(base_dir)
        self.feature_type = feature_type
        self.extractor = FeatureExtractor()
        self.sample_rate = 16000

        # Determine cache directory. If an explicit cache_dir is given, use it.
        # Otherwise, look for a default cache under the dataset root: 
        # <database_root>/features/feat{feature_type}/{split}/
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is None:
            # try to infer database root and split from base_dir
            try:
                db_root = self.base_dir.parents[1]
            except Exception:
                db_root = self.base_dir
            base_name = str(self.base_dir).lower()
            if "train" in base_name:
                split = "train"
            elif "dev" in base_name:
                split = "dev"
            elif "eval" in base_name:
                split = "eval"
            else:
                split = "misc"

            default_cache = db_root / "features" / f"feat{self.feature_type}" / split
            if default_cache.exists():
                self.cache_dir = default_cache
            else:
                self.cache_dir = None

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        # If cache is enabled and feature npy exists, load it
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{utt_id}.npy"
            if cache_path.exists():
                try:
                    feature = np.load(str(cache_path), allow_pickle=False)
                    return torch.FloatTensor(feature), self.labels[utt_id]
                except Exception as e:
                    print(f"Error loading feature cache {cache_path}: {e}")

        # Try multiple path variations for audio
        possible_paths = [
            self.base_dir / f"{utt_id}.flac",
            self.base_dir / "flac" / f"{utt_id}.flac",
            self.base_dir / f"{utt_id}.wav",
            self.base_dir / "flac" / f"{utt_id}.wav",
        ]

        filepath = None
        for path in possible_paths:
            if path.exists():
                filepath = path
                break

        if filepath is None:
            print(f"Warning: File not found for {utt_id}. Tried: {possible_paths[0]}")
            audio = np.zeros(64600)
        else:
            try:
                audio, sr = librosa.load(str(filepath), sr=self.sample_rate)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                audio = np.zeros(64600)

        feature = self.extractor.extract_feature(audio, self.feature_type)
        feature = torch.FloatTensor(feature)

        # Save to cache for future runs (best-effort)
        if self.cache_dir is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(self.cache_dir / f"{utt_id}.npy"), feature.numpy(), allow_pickle=False)
            except Exception as e:
                print(f"Warning: could not save feature cache for {utt_id}: {e}")

        label = self.labels[utt_id]
        return feature, label


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, feature_type=0, cache_dir=None):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.feature_type = feature_type
        self.extractor = FeatureExtractor()
        self.sample_rate = 16000

        # Determine cache directory similarly to train dataset
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is None:
            try:
                db_root = self.base_dir.parents[1]
            except Exception:
                db_root = self.base_dir
            base_name = str(self.base_dir).lower()
            if "train" in base_name:
                split = "train"
            elif "dev" in base_name:
                split = "dev"
            elif "eval" in base_name:
                split = "eval"
            else:
                split = "misc"
            default_cache = db_root / "features" / f"feat{self.feature_type}" / split
            if default_cache.exists():
                self.cache_dir = default_cache
            else:
                self.cache_dir = None

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        # If cache is enabled and feature npy exists, load it
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{utt_id}.npy"
            if cache_path.exists():
                try:
                    feature = np.load(str(cache_path), allow_pickle=False)
                    return torch.FloatTensor(feature), utt_id
                except Exception as e:
                    print(f"Error loading feature cache {cache_path}: {e}")

        # Try multiple path variations for audio
        possible_paths = [
            self.base_dir / f"{utt_id}.flac",
            self.base_dir / "flac" / f"{utt_id}.flac",
            self.base_dir / f"{utt_id}.wav",
            self.base_dir / "flac" / f"{utt_id}.wav",
        ]

        filepath = None
        for path in possible_paths:
            if path.exists():
                filepath = path
                break

        if filepath is None:
            print(f"Warning: File not found for {utt_id}. Tried: {possible_paths[0]}")
            audio = np.zeros(64600)
        else:
            try:
                audio, sr = librosa.load(str(filepath), sr=self.sample_rate)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                audio = np.zeros(64600)

        feature = self.extractor.extract_feature(audio, self.feature_type)
        feature = torch.FloatTensor(feature)

        # Save to cache for future runs (best-effort)
        if self.cache_dir is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(str(self.cache_dir / f"{utt_id}.npy"), feature.numpy(), allow_pickle=False)
            except Exception as e:
                print(f"Warning: could not save feature cache for {utt_id}: {e}")

        return feature, utt_id
