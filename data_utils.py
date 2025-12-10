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
    4: "cqt",  # Constant-Q Transform - best for fake vs real audio
}


def add_random_noise(waveform: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add random Gaussian noise to waveform.
    
    Args:
        waveform: Audio waveform as numpy array
        snr_db: Signal-to-Noise Ratio in dB (higher = less noise)
    
    Returns:
        Noisy waveform
    """
    signal_power = np.mean(waveform ** 2)
    
    # Calculate noise power based on SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate random Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    
    # Add noise to signal
    noisy_waveform = waveform + noise
    
    return noisy_waveform


def add_background_noise(waveform: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
    """
    Add synthetic background noise (e.g., white noise, pink noise).
    
    Args:
        waveform: Audio waveform as numpy array
        noise_factor: Noise amplitude factor (0-1)
    
    Returns:
        Waveform with added background noise
    """
    # Generate white noise
    noise = np.random.normal(0, 1, waveform.shape)
    
    # Normalize and scale
    noise = noise / np.max(np.abs(noise)) if np.max(np.abs(noise)) > 0 else noise
    noise = noise * noise_factor * np.max(np.abs(waveform))
    
    return waveform + noise


def add_reverberation(waveform: np.ndarray, reverb_factor: float = 0.5) -> np.ndarray:
    """
    Add simple reverberation effect (echo).
    
    Args:
        waveform: Audio waveform as numpy array
        reverb_factor: Echo amplitude factor (0-1)
    
    Returns:
        Waveform with reverberation
    """
    # Create a simple echo by delaying and adding
    delay_samples = int(0.05 * 16000)  # 50ms delay at 16kHz
    
    if len(waveform) <= delay_samples:
        return waveform
    
    reverb = waveform.copy()
    reverb[delay_samples:] += reverb_factor * waveform[:-delay_samples]
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(reverb))
    if max_val > 1.0:
        reverb = reverb / max_val
    
    return reverb


def add_pitch_shift(waveform: np.ndarray, semitones: float = 2.0, sr: int = 16000) -> np.ndarray:
    """
    Apply pitch shifting using librosa.
    
    Args:
        waveform: Audio waveform as numpy array
        semitones: Number of semitones to shift
        sr: Sample rate
    
    Returns:
        Pitch-shifted waveform
    """
    if not LIBROSA_AVAILABLE:
        return waveform
    
    try:
        shifted = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=int(semitones))
        return shifted
    except Exception:
        return waveform


def apply_augmentation(waveform: np.ndarray, augmentation_type: int = 0, sr: int = 16000) -> np.ndarray:
    """
    Apply various augmentations to waveform.
    
    Args:
        waveform: Audio waveform as numpy array
        augmentation_type: 0=no_aug, 1=gaussian_noise, 2=background_noise, 
                          3=reverberation, 4=pitch_shift
        sr: Sample rate
    
    Returns:
        Augmented waveform
    """
    if augmentation_type == 0:
        return waveform
    elif augmentation_type == 1:
        # Gaussian noise with random SNR (15-30 dB)
        snr = np.random.uniform(15, 30)
        return add_random_noise(waveform, snr_db=snr)
    elif augmentation_type == 2:
        # Background noise with random factor (0.005-0.02)
        factor = np.random.uniform(0.005, 0.02)
        return add_background_noise(waveform, noise_factor=factor)
    elif augmentation_type == 3:
        # Reverberation with random factor (0.3-0.7)
        factor = np.random.uniform(0.3, 0.7)
        return add_reverberation(waveform, reverb_factor=factor)
    elif augmentation_type == 4:
        # Pitch shift with random semitones (-2 to +2)
        semitones = np.random.uniform(-2, 2)
        return add_pitch_shift(waveform, semitones=semitones, sr=sr)
    else:
        return waveform


def extract_feature(waveform: np.ndarray, feature_type: int = 0, sr: int = 16000):
    """
    Extract different features from waveform.
    
    Args:
        waveform: Audio waveform as numpy array
        feature_type: 0=raw, 1=mel_spectrogram, 2=lfcc, 3=mfcc, 4=cqt
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
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    elif feature_type == 2:
        # ==========================
        # Linear Frequency Cepstral Coefficients (LFCC)
        # ==========================
        import numpy as np
        from scipy.fftpack import dct

        # 1. STFT → power spectrum
        S = np.abs(librosa.stft(y=waveform, n_fft=512, hop_length=160)) ** 2

        # 2. Linear filterbank
        # librosa already supports linear filters
        fb = librosa.filters.linear(
            sr=sr,
            n_fft=512,
            n_filters=20  # typical LFCC config
        )

        # 3. Apply filterbank
        S_lin = np.dot(fb, S)

        # 4. Log
        log_S = np.log(S_lin + 1e-6)

        # 5. DCT → LFCC (first 13 coefficients)
        lfcc = dct(log_S, type=2, axis=0, norm='ortho')[:13]

        return lfcc
    
    elif feature_type == 3:
        # Mel-Frequency Cepstral Coefficients (MFCC)
        mfcc = librosa.feature.mfcc(
            y=waveform, sr=sr, n_mfcc=13, n_fft=512, hop_length=160
        )
        return mfcc
    
    elif feature_type == 4:
        # Constant-Q Transform (CQT)
        cqt = librosa.cqt(
            y=waveform, sr=sr, hop_length=512, n_bins=84, bins_per_octave=12
        )
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db
    
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
    def __init__(self, list_IDs, labels, base_dir, feature_type: int = 0, sr: int = 16000, random_noise: bool = False):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)
           self.feature_type: type of feature to extract (0=raw, 1=mel_spec, 2=lfcc, 3=mfcc, 4=cqt)
           self.sr          : sample rate
           self.random_noise: whether to apply random augmentation"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.feature_type = feature_type
        self.sr = sr
        self.random_noise = random_noise
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, sr = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        
        # Apply random augmentation if enabled (only for training)
        if self.random_noise:
            # Randomly select augmentation type or no augmentation
            aug_type = np.random.randint(0, 5)  # 0-4: no_aug, gaussian, bg_noise, reverb, pitch_shift
            X = apply_augmentation(X, augmentation_type=aug_type, sr=sr)
        
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
        """self.list_IDs\t: list of strings (each string: utt key),
           self.feature_type: type of feature to extract (0=raw, 1=mel_spec, 2=lfcc, 3=mfcc, 4=cqt)
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
