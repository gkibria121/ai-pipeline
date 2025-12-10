"""
Dataset factory for creating appropriate dataset loaders based on dataset type.
Supports ASVspoof2019, Fake-or-Real, and SceneFake datasets.
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import soundfile as sf
from torch import Tensor
from torch.utils.data import Dataset

from data_utils import extract_feature, apply_augmentation, pad, pad_random


# Dataset type mappings
DATASET_TYPES = {
    1: "ASVspoof2019",
    2: "Fake-or-Real", 
    3: "SceneFake"
}


def get_dataset_info(dataset_type: int) -> Dict:
    """
    Get dataset information including paths and structure.
    
    Args:
        dataset_type: 1=ASVspoof2019, 2=Fake-or-Real, 3=SceneFake
        
    Returns:
        Dictionary with dataset configuration
    """
    if dataset_type == 1:
        return {
            "name": "ASVspoof2019",
            "base_path": "./LA",
            "has_protocols": True,
            "track": "LA",
            "file_format": "flac"
        }
    elif dataset_type == 2:
        return {
            "name": "Fake-or-Real",
            "base_path": "./fake_or_real/for-2sec/for-2seconds",
            "has_protocols": False,
            "track": None,
            "file_format": "wav"
        }
    elif dataset_type == 3:
        return {
            "name": "SceneFake",
            "base_path": "./scenefake",
            "has_protocols": False,
            "track": None,
            "file_format": "wav"
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


class Dataset_FakeOrReal_train(Dataset):
    """Dataset for Fake-or-Real training set."""
    
    def __init__(self, list_IDs, labels, base_dir, feature_type: int = 0, 
                 sr: int = 16000, random_noise: bool = False):
        """
        Args:
            list_IDs: list of file identifiers
            labels: dictionary mapping file IDs to labels (1=real, 0=fake)
            base_dir: base directory containing audio files
            feature_type: type of feature to extract
            sr: sample rate
            random_noise: whether to apply random augmentation
        """
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = Path(base_dir)
        self.feature_type = feature_type
        self.sr = sr
        self.random_noise = random_noise
        self.cut = 64600  # ~4 sec audio at 16kHz
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # Load audio file
        audio_path = self.base_dir / key
        X, sr = sf.read(str(audio_path))
        
        # Apply random augmentation if enabled
        if self.random_noise:
            aug_type = np.random.randint(0, 5)
            X = apply_augmentation(X, aug_type, sr)
        
        # Extract features
        X_feat = extract_feature(X, feature_type=self.feature_type, sr=sr)
        
        # Apply padding based on feature type
        if self.feature_type == 0:
            X_pad = pad_random(X_feat, self.cut)
            x_inp = Tensor(X_pad)
        else:
            # For time-frequency features
            time_steps = X_feat.shape[1]
            target_steps = int(self.cut / 160) + 1
            
            if time_steps >= target_steps:
                # Use CENTER cropping for deterministic evaluation (not random!)
                stt = (time_steps - target_steps) // 2
                X_pad = X_feat[:, stt:stt + target_steps]
            else:
                num_repeats = int(target_steps / time_steps) + 1
                X_pad = np.tile(X_feat, (1, num_repeats))[:, :target_steps]
            
            x_inp = Tensor(X_pad)
        
        y = self.labels[key]
        return x_inp, y


class Dataset_FakeOrReal_devNeval(Dataset):
    """Dataset for Fake-or-Real dev/eval set."""
    
    def __init__(self, list_IDs, base_dir, feature_type: int = 0, sr: int = 16000):
        """
        Args:
            list_IDs: list of file identifiers
            base_dir: base directory containing audio files
            feature_type: type of feature to extract
            sr: sample rate
        """
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.feature_type = feature_type
        self.sr = sr
        self.cut = 64600
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        key = self.list_IDs[index]
        
        # Load audio file
        audio_path = self.base_dir / key
        X, sr = sf.read(str(audio_path))
        
        # Extract features
        X_feat = extract_feature(X, feature_type=self.feature_type, sr=sr)
        
        # Apply padding - use deterministic center cropping for evaluation
        if self.feature_type == 0:
            X_pad = pad(X_feat, self.cut)
            x_inp = Tensor(X_pad)
        else:
            time_steps = X_feat.shape[1]
            target_steps = int(self.cut / 160) + 1
            
            if time_steps >= target_steps:
                # Use CENTER cropping for deterministic evaluation (not random!)
                stt = (time_steps - target_steps) // 2
                X_pad = X_feat[:, stt:stt + target_steps]
            else:
                num_repeats = int(target_steps / time_steps) + 1
                X_pad = np.tile(X_feat, (1, num_repeats))[:, :target_steps]
            
            x_inp = Tensor(X_pad)
        
        return x_inp, key


def load_fake_or_real_data(base_path: Path) -> Tuple[Dict, List, Dict, List, List]:
    """
    Load Fake-or-Real dataset file lists and labels.
    
    Returns:
        train_labels, train_files, dev_labels, dev_files, eval_files
    """
    base_path = Path(base_path)
    
    # Structure: training/{real,fake}/, testing/{real,fake}/, validation/{real,fake}/
    train_real_dir = base_path / "training" / "real"
    train_fake_dir = base_path / "training" / "fake"
    test_real_dir = base_path / "testing" / "real"
    test_fake_dir = base_path / "testing" / "fake"
    val_real_dir = base_path / "validation" / "real"
    val_fake_dir = base_path / "validation" / "fake"
    
    train_labels = {}
    train_files = []
    dev_labels = {}
    dev_files = []
    eval_labels = {}
    eval_files = []
    
    # Load training real files
    if train_real_dir.exists():
        for audio_file in train_real_dir.glob("*.wav"):
            rel_path = f"training/real/{audio_file.name}"
            train_files.append(rel_path)
            train_labels[rel_path] = 1  # 1 = bonafide/real
    
    # Load training fake files
    if train_fake_dir.exists():
        for audio_file in train_fake_dir.glob("*.wav"):
            rel_path = f"training/fake/{audio_file.name}"
            train_files.append(rel_path)
            train_labels[rel_path] = 0  # 0 = spoof/fake
    
    # Load validation real files (use as dev set)
    if val_real_dir.exists():
        for audio_file in val_real_dir.glob("*.wav"):
            rel_path = f"validation/real/{audio_file.name}"
            dev_files.append(rel_path)
            dev_labels[rel_path] = 1
    
    # Load validation fake files (use as dev set)
    if val_fake_dir.exists():
        for audio_file in val_fake_dir.glob("*.wav"):
            rel_path = f"validation/fake/{audio_file.name}"
            dev_files.append(rel_path)
            dev_labels[rel_path] = 0
    
    # Load testing real files (use as eval set)
    if test_real_dir.exists():
        for audio_file in test_real_dir.glob("*.wav"):
            rel_path = f"testing/real/{audio_file.name}"
            eval_files.append(rel_path)
            eval_labels[rel_path] = 1
    
    # Load testing fake files (use as eval set)
    if test_fake_dir.exists():
        for audio_file in test_fake_dir.glob("*.wav"):
            rel_path = f"testing/fake/{audio_file.name}"
            eval_files.append(rel_path)
            eval_labels[rel_path] = 0
    
    # Print dataset statistics
    print(f"\nDataset loaded from: {base_path}")
    print(f"Training samples: {len(train_files)} (Real: {sum(1 for v in train_labels.values() if v == 1)}, Fake: {sum(1 for v in train_labels.values() if v == 0)})")
    print(f"Validation samples: {len(dev_files)} (Real: {sum(1 for v in dev_labels.values() if v == 1)}, Fake: {sum(1 for v in dev_labels.values() if v == 0)})")
    print(f"Testing samples: {len(eval_files)} (Real: {sum(1 for v in eval_labels.values() if v == 1)}, Fake: {sum(1 for v in eval_labels.values() if v == 0)})\n")
    
    if len(train_files) == 0:
        raise ValueError(f"No training files found in {base_path}. Please check the dataset path and structure.")
    
    return train_labels, train_files, dev_labels, dev_files, eval_files


def create_dataset_loaders(dataset_type: int, base_path: Path, feature_type: int, 
                           random_noise: bool, batch_size: int, seed: int):
    """
    Create appropriate dataset loaders based on dataset type.
    
    Args:
        dataset_type: 1=ASVspoof2019, 2=Fake-or-Real, 3=SceneFake
        base_path: Base path to dataset
        feature_type: Feature type to extract
        random_noise: Whether to apply augmentation
        batch_size: Batch size for dataloaders
        seed: Random seed
        
    Returns:
        train_loader, dev_loader, eval_loader
    """
    import torch
    from torch.utils.data import DataLoader
    from utils import seed_worker
    
    if dataset_type == 2:  # Fake-or-Real
        train_labels, train_files, dev_labels, dev_files, eval_files = load_fake_or_real_data(base_path)
        
        train_set = Dataset_FakeOrReal_train(
            list_IDs=train_files,
            labels=train_labels,
            base_dir=base_path,
            feature_type=feature_type,
            random_noise=random_noise
        )
        
        dev_set = Dataset_FakeOrReal_devNeval(
            list_IDs=dev_files,
            base_dir=base_path,
            feature_type=feature_type
        )
        
        eval_set = Dataset_FakeOrReal_devNeval(
            list_IDs=eval_files,
            base_dir=base_path,
            feature_type=feature_type
        )
        
        gen = torch.Generator()
        gen.manual_seed(seed)
        
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=gen
        )
        
        dev_loader = DataLoader(
            dev_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        
        eval_loader = DataLoader(
            eval_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
        
        return train_loader, dev_loader, eval_loader
    
    else:
        raise NotImplementedError(f"Dataset type {dataset_type} not yet implemented in factory")
