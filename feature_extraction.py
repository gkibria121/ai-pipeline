"""
Feature extraction utilities for audio processing
Supports: Raw Audio, Mel Spectrogram, MFCC, LFCC, CQT
"""

import numpy as np
import librosa
import torch


class FeatureExtractor:
    """Feature extraction module for different audio representations"""
    
    def __init__(self, sample_rate=16000, n_fft=512, n_mels=128, n_mfcc=13, n_lfcc=13):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc
    
    def extract_raw_audio(self, audio):
        """Feature 0: Raw audio (no processing)"""
        return audio
    
    def extract_mel_spectrogram(self, audio):
        """Feature 1: Mel Spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.n_fft//4, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel.astype(np.float32)
    
    def extract_mfcc(self, audio):
        """Feature 2: MFCC (Mel-Frequency Cepstral Coefficients)"""
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.n_fft//4, n_mfcc=self.n_mfcc, n_mels=self.n_mels)
        return mfcc.astype(np.float32)
    
    def extract_lfcc(self, audio):
        """Feature 3: LFCC (Linear-Frequency Cepstral Coefficients)"""
        # Compute power spectrogram
        spec = np.abs(librosa.stft(
            audio, n_fft=self.n_fft, 
            hop_length=self.n_fft//4)) ** 2
        
        # Apply DCT-like transformation for cepstral coefficients
        lfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(spec), n_mfcc=self.n_lfcc)
        return lfcc.astype(np.float32)
    
    def extract_cqt(self, audio):
        """Feature 4: CQT (Constant-Q Transform)"""
        cqt = np.abs(librosa.cqt(
            audio, sr=self.sr, 
            hop_length=self.n_fft//4, n_bins=84, bins_per_octave=12))
        log_cqt = librosa.power_to_db(cqt, ref=np.max)
        return log_cqt.astype(np.float32)
    
    def extract_feature(self, audio, feature_type=0):
        """
        Extract specified feature type
        
        Args:
            audio: input audio signal
            feature_type: 0=raw, 1=mel_spec, 2=mfcc, 3=lfcc, 4=cqt
        
        Returns:
            extracted feature
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