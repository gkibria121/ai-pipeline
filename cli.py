import argparse
from typing import Namespace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--dataset",
                        type=int,
                        default=None,
                        choices=[1, 2, 3],
                        help="dataset to use: 1=ASVspoof2019, 2=Fake-or-Real, 3=SceneFake (default: None, uses config value or 1)")
    parser.add_argument("--dataset_version",
                        type=int,
                        default=None,
                        help="(Fake-or-Real only) dataset version identifier. If omitted, defaults to 3 when base path matches known version paths.")
    parser.add_argument("--epochs",
                        type=int,
                        default=None,
                        help="number of epochs to override config (default: None, uses config value)")
    parser.add_argument("--batch_size",
                        type=int,
                        default=None,
                        help="batch size to override config (default: None, uses config value)")
    parser.add_argument("--feature_type",
                        type=int,
                        default=None,
                        choices=[0, 1, 2, 3, 4],
                        help="feature type: 0=raw, 1=mel_spectrogram, 2=lfcc, 3=mfcc, 4=cqt (default: None, uses config value)")
    parser.add_argument("--feature_analysis",
                        action="store_true",
                        help="generate feature analysis visualization before training (disabled by default)")
    parser.add_argument("--random_noise",
                        action="store_true",
                        help="enable random data augmentation (RIR, MUSAN-style noise, pitch shift, time stretch, SpecAugment)")
    parser.add_argument("--weight_avg",
                        action="store_true",
                        help="enable Stochastic Weight Averaging (SWA) for better generalization")
    parser.add_argument("--data_subset",
                        type=float,
                        default=1.0,
                        help="percentage of data to use from each split (0.0-1.0, default: 1.0 for full dataset)")
    parser.add_argument("--eval_best",
                        action="store_true",
                        help="evaluate on test set whenever a new best model is found during training")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="path to the model weight file (can be also given in the config file)")
    parser.add_argument("--cpu",
                        action="store_true",
                        help="force CPU mode even if GPU is available (slower but works without CUDA)")
    return parser


def parse_args(argv=None) -> Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args
