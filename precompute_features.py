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
import torch
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

    # create FeatureExtractor early to get sample length and transforms
    extractor = FeatureExtractor(sample_rate=sr, use_gpu=_CLI_USE_GPU, use_torch=_CLI_USE_TORCH)
    # compute sample length used for padding/truncation
    hop = extractor.n_fft // 4
    nb_samp = extractor.target_frames * hop

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

        # Process files in batches when using torchaudio+GPU for speed
        if extractor.use_torch and extractor.use_gpu:
            # gather ids into chunks
            batch_size = _CLI_BATCH_SIZE
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                # load audios for batch
                audios = []
                save_pairs = []  # (utt_id, cache_path)
                for utt_id in batch_ids:
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
                        audio = np.zeros(nb_samp)
                    else:
                        try:
                            audio, _ = librosa.load(str(filepath), sr=sr)
                        except Exception:
                            audio = np.zeros(nb_samp)
                    # pad/truncate to nb_samp
                    if len(audio) < nb_samp:
                        pad = np.zeros(nb_samp, dtype=audio.dtype)
                        pad[:len(audio)] = audio
                        audio = pad
                    elif len(audio) > nb_samp:
                        audio = audio[:nb_samp]
                    audios.append(audio.astype(np.float32))
                    save_pairs.append((utt_id, cache_path))

                if not audios:
                    continue

                # create batch tensor and move to device
                wav_batch = torch.from_numpy(np.stack(audios, axis=0)).float().to(extractor.device)
                if wav_batch.dim() == 2:
                    wav_batch = wav_batch.unsqueeze(1)  # (B,1,N)

                with torch.no_grad():
                    if feature_type == 1:
                        mel = extractor.ta_mel(wav_batch)
                        db = extractor.ta_db(mel)
                        # db shape: (B, n_mels, time)
                        db = db.cpu().numpy()
                        for (utt_id, cache_path), feat_arr in zip(save_pairs, db):
                            # pad/truncate time dim if needed and apply CMVN per sample
                            feat_proc = extractor._pad_truncate_time(feat_arr)
                            feat_proc = extractor._apply_cmvn(feat_proc)
                            try:
                                np.save(str(cache_path), feat_proc, allow_pickle=False)
                            except Exception:
                                pass
                    elif feature_type == 2 and getattr(extractor, 'ta_mfcc', None) is not None:
                        mfcc = extractor.ta_mfcc(wav_batch)
                        mfcc = mfcc.cpu().numpy()
                        for (utt_id, cache_path), feat_arr in zip(save_pairs, mfcc):
                            feat_proc = extractor._pad_truncate_time(feat_arr)
                            try:
                                np.save(str(cache_path), feat_proc, allow_pickle=False)
                            except Exception:
                                pass
                    else:
                        # fallback: compute per sample on CPU if other features
                        for (utt_id, cache_path), audio in zip(save_pairs, audios):
                            feat = extractor.extract_feature(audio, feature_type)
                            try:
                                np.save(str(cache_path), feat, allow_pickle=False)
                            except Exception:
                                pass
        else:
            # CPU or torchaudio not available: per-file processing
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
                    audio = np.zeros(nb_samp)
                else:
                    try:
                        audio, _ = librosa.load(str(filepath), sr=sr)
                    except Exception:
                        audio = np.zeros(nb_samp)

                feat = extractor.extract_feature(audio, feature_type)
                try:
                    np.save(str(cache_path), feat, allow_pickle=False)
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="LA", help="Root path to dataset (default: LA)")
    parser.add_argument("--track", default="LA", help="LA, PA or DF")
    parser.add_argument("--feature_type", type=int, default=1, help="Feature type (0=raw,1=mel,2=mfcc,3=lfcc,4=cqt,5=mel+delta)")
    # Splits are fixed: train, dev, eval
    parser.add_argument("--out_root", default=None, help="Optional output root for cache (defaults to <database_path>/features)")
    parser.add_argument("--features_path", default=None, help="Alias for --out_root (explicit writable path for cached features)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for GPU processing (default: 256)")
    args = parser.parse_args()

    # features_path takes precedence when provided (convenience alias)
    chosen_out = args.features_path if args.features_path is not None else args.out_root
    out_root = Path(chosen_out) if chosen_out else None

    # Always attempt to use torchaudio + GPU when available (no flags needed)
    global _CLI_USE_GPU, _CLI_USE_TORCH
    _CLI_USE_TORCH = True
    _CLI_USE_GPU = True
    # batch size for GPU batching
    global _CLI_BATCH_SIZE
    _CLI_BATCH_SIZE = int(args.batch_size)

    # Fixed splits
    splits = ["train", "dev", "eval"]

    precompute(Path(args.database_path), args.track, args.feature_type, splits, out_root=out_root)
