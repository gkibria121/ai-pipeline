"""
Dataset download script for ASVspoof and other deepfake audio datasets.

Usage:
    python download_dataset.py --dataset 1  # Download ASVspoof 2019
    python download_dataset.py --dataset 2  # Download Fake-or-Real dataset
    python download_dataset.py --dataset 3  # Download SceneFake dataset
"""

import argparse
import os
import shutil
import kagglehub


def download_asvspoof2019():
    """Download ASVspoof 2019 dataset."""
    print("=" * 70)
    print("Downloading ASVspoof 2019 Dataset")
    print("=" * 70)
    
    awsaf49_asvpoof_2019_dataset_path = kagglehub.dataset_download('awsaf49/asvpoof-2019-dataset')
    print('[OK] Data source download complete.')
    
    src = os.path.join(awsaf49_asvpoof_2019_dataset_path, "LA", "LA")  # real folder
    dst = "./LA"  # directory to create in current directory
    
    # Remove old symlink or folder if exists
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            print(f"Warning: {dst} exists as a directory. Removing...")
            shutil.rmtree(dst)
    
    # Copy directory (Windows compatible - no symlinks needed)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[OK] Dataset copied: {dst} <- {src}")
    print("[OK] ASVspoof 2019 dataset ready!")


def download_fake_or_real():
    """Download Fake-or-Real dataset."""
    print("=" * 70)
    print("Downloading Fake-or-Real Dataset")
    print("=" * 70)
    
    mohammedabdeldayem_the_fake_or_real_dataset_path = kagglehub.dataset_download(
        'mohammedabdeldayem/the-fake-or-real-dataset'
    )
    print('[OK] Data source download complete.')
    
    src = mohammedabdeldayem_the_fake_or_real_dataset_path
    dst = "./fake_or_real"
    
    # Remove old symlink or folder if exists
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            print(f"Warning: {dst} exists as a directory. Removing...")
            shutil.rmtree(dst)
    
    # Copy directory (Windows compatible - no symlinks needed)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[OK] Dataset copied: {dst} <- {src}")
    print("[OK] Fake-or-Real dataset ready!")


def download_scenefake():
    """Download SceneFake dataset."""
    print("=" * 70)
    print("Downloading SceneFake Dataset")
    print("=" * 70)
    
    mohammedabdeldayem_scenefake_path = kagglehub.dataset_download(
        'mohammedabdeldayem/scenefake'
    )
    print('[OK] Data source download complete.')
    
    src = mohammedabdeldayem_scenefake_path
    dst = "./scenefake"
    
    # Remove old symlink or folder if exists
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            print(f"Warning: {dst} exists as a directory. Removing...")
            shutil.rmtree(dst)
    
    # Copy directory (Windows compatible - no symlinks needed)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"[OK] Dataset copied: {dst} <- {src}")
    print("[OK] SceneFake dataset ready!")


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for audio deepfake detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Options:
  1 - ASVspoof 2019 (awsaf49/asvpoof-2019-dataset)
      Standard benchmark for audio spoofing detection
      
  2 - Fake-or-Real (mohammedabdeldayem/the-fake-or-real-dataset)
      Binary classification dataset for fake vs real audio
      
  3 - SceneFake (mohammedabdeldayem/scenefake)
      Scene-aware fake audio detection dataset

Examples:
  python download_dataset.py --dataset 1    # Download ASVspoof 2019
  python download_dataset.py --dataset 2    # Download Fake-or-Real
  python download_dataset.py --dataset 3    # Download SceneFake
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='Dataset to download: 1=ASVspoof2019, 2=Fake-or-Real, 3=SceneFake'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("DATASET DOWNLOADER")
    print("=" * 70 + "\n")
    
    if args.dataset == 1:
        download_asvspoof2019()
    elif args.dataset == 2:
        download_fake_or_real()
    elif args.dataset == 3:
        download_scenefake()
    
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()