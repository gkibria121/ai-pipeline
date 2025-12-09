"""
Precompute features for ASVspoof2019 dataset and save to cache.
Cache layout created:
  <database_root>/features/feat{feature_type}/{split}/{utt_id}.npy

Usage example:
python precompute_features.py --database_path /path/to/ASV --track LA --feature_type 1 --splits train dev eval

"""
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
from feature_extraction import FeatureExtractor
from data_utils import genSpoof_list
import sys
import errno
import warnings

# Ignore non-critical warnings (user requested)
warnings.filterwarnings("ignore")


def collect_ids(protocol_path: Path, split: str):
    if split == "train":
        d_meta, file_list = genSpoof_list(str(protocol_path), is_train=True)
        return file_list, d_meta
    else:
        file_list = genSpoof_list(str(protocol_path), is_eval=True)
        return file_list, None


def precompute(database_path: Path, track: str, feature_type: int, splits, sr=16000, out_root: Path = None):
    prefix_2019 = f"ASVspoof2019.{track}"
    database_path = Path(database_path)
    if out_root is None:
        out_root = database_path / "features"

    # create FeatureExtractor with defaults (can be overridden via CLI)
    extractor = None

    for split in splits:
        if split == "train":
            protocol = database_path / f"ASVspoof2019_{track}_cm_protocols" / f"{prefix_2019}.cm.train.trn.txt"
            ids, labels = collect_ids(protocol, "train")
            base_dir = database_path / f"ASVspoof2019_{track}_train" / "flac"
        elif split == "dev":
            protocol = database_path / f"ASVspoof2019_{track}_cm_protocols" / f"{prefix_2019}.cm.dev.trl.txt"
            ids, _ = collect_ids(protocol, "dev")
            base_dir = database_path / f"ASVspoof2019_{track}_dev" / "flac"
        elif split == "eval":
            protocol = database_path / f"ASVspoof2019_{track}_cm_protocols" / f"{prefix_2019}.cm.eval.trl.txt"
            ids, _ = collect_ids(protocol, "eval")
            base_dir = database_path / f"ASVspoof2019_{track}_eval" / "flac"
        else:
            print(f"Unknown split: {split}")
            continue

        cache_dir = out_root / f"feat{feature_type}" / split
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EROFS, errno.EPERM) or getattr(e, 'errno', None) == 30:
                print(f"Error: cannot create cache directory '{cache_dir}'. filesystem may be read-only or permission denied.")
                print("Use the --features_path (or --out_root) flag to specify a writable location for cached features.")
                sys.exit(1)
            else:
                raise

        print(f"Precomputing features for {split} -> {cache_dir} ({len(ids)} files)")

        for utt_id in tqdm(ids):
            cache_path = cache_dir / f"{utt_id}.npy"
            if cache_path.exists():
                continue

            # try multiple audio path variants
            possible_paths = [
                base_dir / f"{utt_id}.flac",
                base_dir / "flac" / f"{utt_id}.flac",
                base_dir / f"{utt_id}.wav",
                base_dir / "flac" / f"{utt_id}.wav",
            ]

            filepath = None
            for p in possible_paths:
                if p.exists():
                    filepath = p
                    break

            if filepath is None:
                # try looking directly under database root
                print(f"Warning: audio for {utt_id} not found under {base_dir}")
                audio = np.zeros(64600)
            else:
                try:
                    audio, _ = librosa.load(str(filepath), sr=sr)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    audio = np.zeros(64600)

            # lazily construct extractor once we know device preferences from args
            if extractor is None:
                # default: do not use GPU unless CLI requested it
                # `use_torch` and `use_gpu` are set from parsed args below
                extractor = FeatureExtractor(sample_rate=sr, use_gpu=_CLI_USE_GPU, use_torch=_CLI_USE_TORCH)

            feat = extractor.extract_feature(audio, feature_type)
            try:
                np.save(str(cache_path), feat, allow_pickle=False)
            except Exception as e:
                print(f"Could not save {cache_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="LA", help="Root path to dataset (default: LA)")
    parser.add_argument("--track", default="LA", help="LA, PA or DF")
    parser.add_argument("--feature_type", type=int, default=1, help="Feature type (0=raw,1=mel,2=mfcc,3=lfcc,4=cqt,5=mel+delta)")
    # Splits are fixed: train, dev, eval
    parser.add_argument("--out_root", default=None, help="Optional output root for cache (defaults to <database_path>/features)")
    parser.add_argument("--features_path", default=None, help="Alias for --out_root (explicit writable path for cached features)")
    args = parser.parse_args()

    # features_path takes precedence when provided (convenience alias)
    chosen_out = args.features_path if args.features_path is not None else args.out_root
    out_root = Path(chosen_out) if chosen_out else None

    # Always attempt to use torchaudio + GPU when available (no flags needed)
    global _CLI_USE_GPU, _CLI_USE_TORCH
    _CLI_USE_TORCH = True
    _CLI_USE_GPU = True

    # Fixed splits
    splits = ["train", "dev", "eval"]

    precompute(Path(args.database_path), args.track, args.feature_type, splits, out_root=out_root)
