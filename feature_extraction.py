"""
Feature extraction utilities for audio processing
Supports: Raw Audio, Mel Spectrogram, MFCC, LFCC, CQT
"""

import numpy as np
import librosa
import random

import torch
from torch.utils.data import Dataset

class FeatureExtractor:
    """Feature extraction module for different audio representations"""
    
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=128, n_mfcc=13, n_lfcc=13, nb_samp=64600, target_frames=None):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc

        # derive target_frames from nb_samp and hop_length if not provided
        hop = self.n_fft // 4
        default_frames = int(np.ceil(nb_samp / hop))
        self.target_frames = int(target_frames) if target_frames is not None else default_frames
        # SpecAugment defaults (disabled by default)
        self.spec_augment = False
        self.freq_mask_param = 15
        self.time_mask_param = 25
        self.num_freq_masks = 2
        self.num_time_masks = 2

    def _pad_truncate_time(self, spec: np.ndarray):
        """
        Ensure spectrogram-like arrays have shape (freq_bins, target_frames).
        If shorter -> zero-pad on the right. If longer -> center-crop.
        """
        if spec.ndim != 2:
            raise ValueError("spec must be 2D (freq, time)")
        f, t = spec.shape
        if t == self.target_frames:
            return spec
        if t < self.target_frames:
            pad_width = self.target_frames - t
            return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=(spec.min(),))
        # t > target_frames -> center crop
        start = max(0, (t - self.target_frames) // 2)
        return spec[:, start:start + self.target_frames]

    def _apply_cmvn(self, spec: np.ndarray, eps=1e-9):
        # spec: (freq, time) or (channels, freq, time)
        if spec.ndim == 2:
            m = spec.mean(axis=1, keepdims=True)
            s = spec.std(axis=1, keepdims=True)
            return (spec - m) / (s + eps)
        else:
            # per-channel, per-frequency normalization
            C, F, T = spec.shape
            spec2 = spec.reshape(C * F, T)
            m = spec2.mean(axis=1, keepdims=True)
            s = spec2.std(axis=1, keepdims=True)
            spec2 = (spec2 - m) / (s + eps)
            return spec2.reshape(C, F, T)

    def extract_raw_audio(self, audio):
        """Feature 0: Raw audio (no processing)"""
        # Ensure fixed length of nb_samp (64600 default)
        nb_samp = self.target_frames * (self.n_fft // 4)  # approx inverse mapping
        if len(audio) < nb_samp:
            audio = np.pad(audio, (0, nb_samp - len(audio)), mode='constant')
        elif len(audio) > nb_samp:
            audio = audio[:nb_samp]
        return audio.astype(np.float32)
    
    def extract_mel_spectrogram(self, audio, augment=False):
        """Feature 1: Mel Spectrogram (padded/truncated to fixed frames)"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.n_fft//4, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = self._pad_truncate_time(log_mel)
        if (augment or getattr(self, 'spec_augment', False)):
            log_mel = self._spec_augment(log_mel)
        log_mel = self._apply_cmvn(log_mel)
        return log_mel.astype(np.float32)

    def _spec_augment(self, spec: np.ndarray):
        """Apply simple SpecAugment (frequency and time masking) in-place copy.

        Args:
            spec: np.ndarray of shape (freq, time)
        Returns:
            augmented spec as np.ndarray
        """
        spec_aug = spec.copy()
        F, T = spec_aug.shape

        # Frequency masks
        for _ in range(self.num_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, F))
            if f == 0:
                continue
            f0 = random.randint(0, max(0, F - f))
            spec_aug[f0:f0 + f, :] = spec_aug.min()

        # Time masks
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T))
            if t == 0:
                continue
            t0 = random.randint(0, max(0, T - t))
            spec_aug[:, t0:t0 + t] = spec_aug.min()

        return spec_aug
    
    def extract_mfcc(self, audio):
        """Feature 2: MFCC (padded/truncated to fixed frames)"""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.n_fft//4, n_mfcc=self.n_mfcc, n_mels=self.n_mels)
        mfcc = self._pad_truncate_time(mfcc)
        return mfcc.astype(np.float32)
    
    def extract_lfcc(self, audio):
        """Feature 3: LFCC (padded/truncated to fixed frames)"""
        spec = np.abs(librosa.stft(
            audio, n_fft=self.n_fft, 
            hop_length=self.n_fft//4)) ** 2
        log_spec = librosa.power_to_db(spec)
        lfcc = librosa.feature.mfcc(
            S=log_spec, n_mfcc=self.n_lfcc)
        lfcc = self._pad_truncate_time(lfcc)
        return lfcc.astype(np.float32)
    
    def extract_cqt(self, audio):
        """Feature 4: CQT (padded/truncated to fixed frames)"""
        cqt = np.abs(librosa.cqt(
            audio, sr=self.sr, 
            hop_length=self.n_fft//4, n_bins=84, bins_per_octave=12))
        log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
        log_cqt = self._pad_truncate_time(log_cqt)
        return log_cqt.astype(np.float32)
    
    def extract_mel_with_delta(self, audio, augment=False):
        """Extract mel spectrogram and its delta and delta-delta"""
        m = self.extract_mel_spectrogram(audio, augment=augment)
        delta = librosa.feature.delta(m)
        delta2 = librosa.feature.delta(m, order=2)
        stacked = np.vstack([m, delta, delta2])  # (3*n_mels, time)
        # reshape to (C, F, T) where C=3, F=n_mels
        C = 3
        stacked = stacked.reshape(C, self.n_mels, -1)
        # pad/truncate in time to target_frames
        stacked = np.stack([self._pad_truncate_time(ch) for ch in stacked])
        stacked = self._apply_cmvn(stacked)
        return stacked.astype(np.float32)
    
    def extract_feature(self, audio, feature_type=0):
        """
        Extract specified feature type
        
        Args:
            audio: input audio signal
            feature_type: 0=raw, 1=mel_spec, 2=mfcc, 3=lfcc, 4=cqt
        
        Returns:
            extracted feature as numpy array
        """
        if feature_type == 0:
            return self.extract_raw_audio(audio)
        elif feature_type == 1:
            return self.extract_mel_spectrogram(audio)
        elif feature_type == 2:
            return self.extract_mfcc(audio)
        elif feature_type == 3:
            return self.extract_lfcc(audio)
        elif feature_type == 4:
            return self.extract_cqt(audio)
        elif feature_type == 5:
            return self.extract_mel_with_delta(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")


# In data_utils.py, update the Dataset classes:

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, feature_type=0):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.feature_type = feature_type
        from feature_extraction import FeatureExtractor
        self.extractor = FeatureExtractor()
        
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        filepath = self.base_dir / f"{utt_id}.flac"
        audio, sr = librosa.load(filepath, sr=16000)
        feature = self.extractor.extract_feature(audio, self.feature_type)
        
        if self.feature_type == 0:  # raw audio
            feature = torch.FloatTensor(feature)
        else:  # spectrogram features
            feature = torch.FloatTensor(feature)
        
        label = self.labels[utt_id]
        return feature, label


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, feature_type=0):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.feature_type = feature_type
        from feature_extraction import FeatureExtractor
        self.extractor = FeatureExtractor()
        
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        filepath = self.base_dir / f"{utt_id}.flac"
        audio, sr = librosa.load(filepath, sr=16000)
        feature = self.extractor.extract_feature(audio, self.feature_type)
        
        if self.feature_type == 0:  # raw audio
            feature = torch.FloatTensor(feature)
        else:  # spectrogram features
            feature = torch.FloatTensor(feature)
        
        return feature, utt_id