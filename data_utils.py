import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    import librosa
    import librosa.feature
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

# Feature type mappings
FEATURE_TYPES = {
    0: "raw",
    1: "mel_spectrogram",
    2: "lfcc",
    3: "mfcc",
}


def extract_feature(waveform: np.ndarray, feature_type: int = 0, sr: int = 16000):
    """
    Extract different features from waveform.
    
    Args:
        waveform: Audio waveform as numpy array
        feature_type: 0=raw, 1=mel_spectrogram, 2=lfcc, 3=mfcc
        sr: Sample rate (default: 16000)
    
    Returns:
        Feature representation as numpy array
    """
    if feature_type == 0:
        # Raw waveform
        return waveform
    
    if not LIBROSA_AVAILABLE:
        raise ImportError(
            "librosa is required for feature extraction. "
            "Install it with: pip install librosa"
        )
    
    if feature_type == 1:
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=128, n_fft=512, hop_length=160
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    elif feature_type == 2:
        # Linear Frequency Cepstral Coefficients (LFCC)
        lfcc = librosa.feature.mfcc(
            y=waveform, sr=sr, n_mfcc=13, n_fft=512, hop_length=160,
            lifter=0  # Use 0 for LFCC (no liftering)
        )
        return lfcc
    
    elif feature_type == 3:
        # Mel-Frequency Cepstral Coefficients (MFCC)
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=sr, n_mfcc=13, n_fft=512, hop_length=160
        )
        return mfcc
    
    else:
        raise ValueError(
            f"Unknown feature_type: {feature_type}. "
            f"Must be one of {list(FEATURE_TYPES.keys())}"
        )


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
    def __init__(self, list_IDs, labels, base_dir, feature_type: int = 0, sr: int = 16000):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)
           self.feature_type: type of feature to extract (0=raw, 1=mel_spec, 2=lfcc, 3=mfcc)
           self.sr          : sample rate"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.feature_type = feature_type
        self.sr = sr
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        
        # Extract features
        X_feat = extract_feature(X, feature_type=self.feature_type, sr=sr)
        
        # Apply padding based on feature type
        if self.feature_type == 0:
            # For raw waveform, use the original padding
            X_pad = pad_random(X_feat, self.cut)
            x_inp = Tensor(X_pad)
        else:
            # For time-frequency features, pad time dimension
            # Shape is (n_features, time_steps)
            time_steps = X_feat.shape[1]
            target_steps = int(self.cut / 160) + 1  # hop_length=160
            
            if time_steps >= target_steps:
                stt = np.random.randint(time_steps - target_steps) if time_steps > target_steps else 0
                X_pad = X_feat[:, stt:stt + target_steps]
            else:
                # Pad if too short
                num_repeats = int(target_steps / time_steps) + 1
                X_pad = np.tile(X_feat, (1, num_repeats))[:, :target_steps]
            
            x_inp = Tensor(X_pad)
        
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, feature_type: int = 0, sr: int = 16000):
        """self.list_IDs	: list of strings (each string: utt key),
           self.feature_type: type of feature to extract (0=raw, 1=mel_spec, 2=lfcc, 3=mfcc)
           self.sr          : sample rate"""
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.feature_type = feature_type
        self.sr = sr
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        
        # Extract features
        X_feat = extract_feature(X, feature_type=self.feature_type, sr=sr)
        
        # Apply padding based on feature type
        if self.feature_type == 0:
            # For raw waveform, use the original padding
            X_pad = pad(X_feat, self.cut)
            x_inp = Tensor(X_pad)
        else:
            # For time-frequency features, pad time dimension
            # Shape is (n_features, time_steps)
            time_steps = X_feat.shape[1]
            target_steps = int(self.cut / 160) + 1  # hop_length=160
            
            if time_steps >= target_steps:
                stt = np.random.randint(time_steps - target_steps) if time_steps > target_steps else 0
                X_pad = X_feat[:, stt:stt + target_steps]
            else:
                # Pad if too short
                num_repeats = int(target_steps / time_steps) + 1
                X_pad = np.tile(X_feat, (1, num_repeats))[:, :target_steps]
            
            x_inp = Tensor(X_pad)
        
        return x_inp, key
