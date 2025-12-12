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


def add_time_stretch(waveform: np.ndarray, rate: float = 1.0, sr: int = 16000) -> np.ndarray:
    """
    Apply time stretching using librosa.
    
    Args:
        waveform: Audio waveform as numpy array
        rate: Stretch factor (>1 = faster, <1 = slower)
        sr: Sample rate
    
    Returns:
        Time-stretched waveform
    """
    if not LIBROSA_AVAILABLE:
        return waveform
    
    try:
        stretched = librosa.effects.time_stretch(waveform, rate=rate)
        return stretched
    except Exception:
        return waveform


def add_gain(waveform: np.ndarray, gain_db: float = 0.0) -> np.ndarray:
    """
    Apply gain (volume change) to waveform.
    
    Args:
        waveform: Audio waveform as numpy array
        gain_db: Gain in decibels (positive = louder, negative = quieter)
    
    Returns:
        Gain-adjusted waveform
    """
    gain_linear = 10 ** (gain_db / 20)
    gained = waveform * gain_linear
    
    # Clip to prevent distortion
    max_val = np.max(np.abs(gained))
    if max_val > 1.0:
        gained = gained / max_val
    
    return gained


def add_low_pass_filter(waveform: np.ndarray, cutoff_freq: float = 4000, sr: int = 16000) -> np.ndarray:
    """
    Apply low-pass filter to simulate phone/low quality audio.
    
    Args:
        waveform: Audio waveform as numpy array
        cutoff_freq: Cutoff frequency in Hz
        sr: Sample rate
    
    Returns:
        Filtered waveform
    """
    try:
        from scipy.signal import butter, filtfilt
        
        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normalized_cutoff, btype='low')
        filtered = filtfilt(b, a, waveform)
        return filtered
    except Exception:
        return waveform


def add_high_pass_filter(waveform: np.ndarray, cutoff_freq: float = 100, sr: int = 16000) -> np.ndarray:
    """
    Apply high-pass filter to remove low frequency noise.
    
    Args:
        waveform: Audio waveform as numpy array
        cutoff_freq: Cutoff frequency in Hz
        sr: Sample rate
    
    Returns:
        Filtered waveform
    """
    try:
        from scipy.signal import butter, filtfilt
        
        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normalized_cutoff, btype='high')
        filtered = filtfilt(b, a, waveform)
        return filtered
    except Exception:
        return waveform


def add_rir_simulation(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulate Room Impulse Response (RIR) effect.
    Creates synthetic room acoustics similar to RIR augmentation.
    
    Args:
        waveform: Audio waveform as numpy array
        sr: Sample rate
    
    Returns:
        Waveform with simulated room acoustics
    """
    try:
        from scipy.signal import fftconvolve
        
        # Generate synthetic RIR (exponentially decaying noise)
        rt60 = np.random.uniform(0.1, 0.5)  # Reverberation time (seconds)
        rir_length = int(rt60 * sr)
        
        # Create exponentially decaying impulse response
        t = np.arange(rir_length) / sr
        decay = np.exp(-3 * t / rt60)  # Exponential decay
        noise = np.random.randn(rir_length)
        rir = noise * decay
        
        # Normalize RIR
        rir = rir / np.max(np.abs(rir))
        
        # Add direct path (stronger initial impulse)
        rir[0] = 1.0
        
        # Convolve with waveform
        convolved = fftconvolve(waveform, rir, mode='same')
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(convolved))
        if max_val > 0:
            convolved = convolved / max_val * np.max(np.abs(waveform))
        
        return convolved
    except Exception:
        return waveform


def add_musan_style_noise(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Add MUSAN-style noise (simulates speech, music, or ambient noise).
    Since we don't have actual MUSAN data, we simulate different noise types.
    
    Args:
        waveform: Audio waveform as numpy array
        sr: Sample rate
    
    Returns:
        Waveform with added noise
    """
    noise_type = np.random.choice(['babble', 'music', 'ambient'])
    
    if noise_type == 'babble':
        # Simulate babble noise (overlapping speech-like sounds)
        # Use filtered noise to simulate speech spectrum
        noise = np.random.randn(len(waveform))
        # Apply bandpass filter (300-3400 Hz, speech range)
        try:
            from scipy.signal import butter, filtfilt
            nyquist = sr / 2
            low = 300 / nyquist
            high = min(3400 / nyquist, 0.99)
            b, a = butter(4, [low, high], btype='band')
            noise = filtfilt(b, a, noise)
        except:
            pass
        snr = np.random.uniform(10, 20)
        
    elif noise_type == 'music':
        # Simulate music-like noise (low frequency emphasis)
        noise = np.random.randn(len(waveform))
        # Apply lowpass filter for music-like spectrum
        try:
            from scipy.signal import butter, filtfilt
            nyquist = sr / 2
            cutoff = min(4000 / nyquist, 0.99)
            b, a = butter(4, cutoff, btype='low')
            noise = filtfilt(b, a, noise)
        except:
            pass
        snr = np.random.uniform(5, 15)
        
    else:  # ambient
        # Ambient noise (broadband with some coloring)
        noise = np.random.randn(len(waveform))
        snr = np.random.uniform(15, 25)
    
    # Calculate scaling factor based on SNR
    signal_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power > 0:
        snr_linear = 10 ** (snr / 10)
        scale = np.sqrt(signal_power / (snr_linear * noise_power))
        noisy = waveform + scale * noise
    else:
        noisy = waveform
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy))
    if max_val > 1.0:
        noisy = noisy / max_val
    
    return noisy


def add_chunking(waveform: np.ndarray, sr: int = 16000, chunk_sec: float = 2.0) -> np.ndarray:
    """
    Apply random chunking - extract a random segment of the audio.
    This helps model learn from different parts of the audio.
    
    Args:
        waveform: Audio waveform as numpy array
        sr: Sample rate
        chunk_sec: Chunk duration in seconds
    
    Returns:
        Random chunk of the waveform (or padded if too short)
    """
    chunk_samples = int(chunk_sec * sr)
    
    if len(waveform) <= chunk_samples:
        # Pad if too short
        padded = np.zeros(chunk_samples)
        padded[:len(waveform)] = waveform
        return padded
    
    # Random start position
    start = np.random.randint(0, len(waveform) - chunk_samples)
    return waveform[start:start + chunk_samples]


def apply_augmentation(waveform: np.ndarray, augmentation_type: int = 0, sr: int = 16000) -> np.ndarray:
    """
    Apply various augmentations to waveform.
    
    Args:
        waveform: Audio waveform as numpy array
        augmentation_type: 0=no_aug, 1=gaussian_noise, 2=background_noise, 
                          3=reverberation, 4=pitch_shift, 5=time_stretch,
                          6=gain, 7=low_pass, 8=high_pass, 9=rir, 10=musan_noise
        sr: Sample rate
    
    Returns:
        Augmented waveform
    """
    if augmentation_type == 0:
        return waveform
    elif augmentation_type == 1:
        # Gaussian noise with random SNR (10-25 dB) - more aggressive
        snr = np.random.uniform(10, 25)
        return add_random_noise(waveform, snr_db=snr)
    elif augmentation_type == 2:
        # Background noise with random factor (0.01-0.05) - more aggressive
        factor = np.random.uniform(0.01, 0.05)
        return add_background_noise(waveform, noise_factor=factor)
    elif augmentation_type == 3:
        # Reverberation with random factor (0.3-0.8)
        factor = np.random.uniform(0.3, 0.8)
        return add_reverberation(waveform, reverb_factor=factor)
    elif augmentation_type == 4:
        # Pitch shift with random semitones (-4 to +4) - wider range
        semitones = np.random.uniform(-4, 4)
        return add_pitch_shift(waveform, semitones=semitones, sr=sr)
    elif augmentation_type == 5:
        # Time stretch (0.85 to 1.15)
        rate = np.random.uniform(0.85, 1.15)
        return add_time_stretch(waveform, rate=rate, sr=sr)
    elif augmentation_type == 6:
        # Gain adjustment (-6 to +6 dB)
        gain_db = np.random.uniform(-6, 6)
        return add_gain(waveform, gain_db=gain_db)
    elif augmentation_type == 7:
        # Low-pass filter (2000-6000 Hz)
        cutoff = np.random.uniform(2000, 6000)
        return add_low_pass_filter(waveform, cutoff_freq=cutoff, sr=sr)
    elif augmentation_type == 8:
        # High-pass filter (50-300 Hz)
        cutoff = np.random.uniform(50, 300)
        return add_high_pass_filter(waveform, cutoff_freq=cutoff, sr=sr)
    elif augmentation_type == 9:
        # RIR simulation (room acoustics)
        return add_rir_simulation(waveform, sr=sr)
    elif augmentation_type == 10:
        # MUSAN-style noise (babble, music, ambient)
        return add_musan_style_noise(waveform, sr=sr)
    else:
        return waveform


def apply_composed_augmentation(waveform: np.ndarray, sr: int = 16000, 
                                 num_augmentations: int = 2, 
                                 augment_prob: float = 0.8) -> np.ndarray:
    """
    Apply multiple random augmentations in sequence for stronger regularization.
    Includes RIR & MUSAN style augmentations similar to top ASVspoof systems.
    
    Args:
        waveform: Audio waveform as numpy array
        sr: Sample rate
        num_augmentations: Number of augmentations to apply (1-3)
        augment_prob: Probability of applying any augmentation
    
    Returns:
        Augmented waveform
    """
    # Skip augmentation with (1 - augment_prob) probability
    if np.random.random() > augment_prob:
        return waveform
    
    # Available augmentation types (excluding 0=no_aug)
    # Includes RIR (9) and MUSAN-style noise (10) for better generalization
    aug_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Randomly select how many augmentations to apply (1 to num_augmentations)
    n_augs = np.random.randint(1, num_augmentations + 1)
    
    # Randomly select which augmentations to apply (without replacement)
    selected_augs = np.random.choice(aug_types, size=min(n_augs, len(aug_types)), replace=False)
    
    # Apply augmentations sequentially
    augmented = waveform.copy()
    for aug_type in selected_augs:
        augmented = apply_augmentation(augmented, augmentation_type=aug_type, sr=sr)
    
    return augmented


def apply_spectrogram_augmentation(spectrogram: np.ndarray, 
                                    freq_mask_prob: float = 0.5,
                                    time_mask_prob: float = 0.5,
                                    max_freq_mask: int = 20,
                                    max_time_mask: int = 50) -> np.ndarray:
    """
    Apply SpecAugment-style augmentation to spectrograms.
    
    Args:
        spectrogram: 2D spectrogram array (freq x time)
        freq_mask_prob: Probability of applying frequency masking
        time_mask_prob: Probability of applying time masking
        max_freq_mask: Maximum frequency bins to mask
        max_time_mask: Maximum time steps to mask
    
    Returns:
        Augmented spectrogram
    """
    spec = spectrogram.copy()
    n_freq, n_time = spec.shape
    
    # Frequency masking (mask random frequency bands)
    if np.random.random() < freq_mask_prob:
        f = np.random.randint(1, min(max_freq_mask, n_freq // 4) + 1)
        f0 = np.random.randint(0, n_freq - f)
        spec[f0:f0 + f, :] = spec.mean()  # Use mean instead of 0 for stability
    
    # Time masking (mask random time segments)
    if np.random.random() < time_mask_prob:
        t = np.random.randint(1, min(max_time_mask, n_time // 4) + 1)
        t0 = np.random.randint(0, n_time - t)
        spec[:, t0:t0 + t] = spec.mean()
    
    return spec


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
        from scipy.fftpack import dct

        n_fft = 512
        hop_length = 160
        n_filters = 20
        n_lfcc = 13

        # 1. STFT → power spectrum
        S = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=hop_length)) ** 2

        # 2. Create linear filterbank manually
        # Linear filterbank with evenly spaced filters in Hz
        freq_bins = n_fft // 2 + 1
        fft_freqs = np.linspace(0, sr / 2, freq_bins)
        
        # Create triangular filters with linear spacing
        filter_freqs = np.linspace(0, sr / 2, n_filters + 2)
        filterbank = np.zeros((n_filters, freq_bins))
        
        for i in range(n_filters):
            # Left, center, right frequencies for triangular filter
            left = filter_freqs[i]
            center = filter_freqs[i + 1]
            right = filter_freqs[i + 2]
            
            # Create triangular filter
            for j, freq in enumerate(fft_freqs):
                if left <= freq <= center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)

        # 3. Apply filterbank
        S_lin = np.dot(filterbank, S)

        # 4. Log (with small epsilon to avoid log(0))
        log_S = np.log(S_lin + 1e-10)

        # 5. DCT → LFCC (first 13 coefficients)
        lfcc = dct(log_S, type=2, axis=0, norm='ortho')[:n_lfcc]

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
            # Use composed augmentation for stronger regularization
            # Apply 1-2 augmentations with 80% probability
            X = apply_composed_augmentation(X, sr=sr, num_augmentations=2, augment_prob=0.8)
        
        # Extract features
        X_feat = extract_feature(X, feature_type=self.feature_type, sr=sr)
        
        # Apply SpecAugment for spectrogram features during training with augmentation
        if self.random_noise and self.feature_type > 0:
            X_feat = apply_spectrogram_augmentation(
                X_feat, 
                freq_mask_prob=0.5, 
                time_mask_prob=0.5,
                max_freq_mask=20,
                max_time_mask=50
            )
        
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
