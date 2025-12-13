# AI-Pipeline (ASVspoof / Anti-Spoofing)

This repository provides a flexible training and evaluation pipeline for audio anti-spoofing research (ASVspoof 2019, Fake-or-Real, SceneFake). It includes model implementations, configs, training utilities, evaluation metrics, and visualization tools.

**Highlights**

- Models: LCNN (and LCNN Large), RawNet3, EfficientNet-B2 (with attention), SEResNet, SimpleCNN, and others in `models/`.
- Config-driven experiments: see `config/` for ready-to-run setups.
- Entry points: `main.py` (train/eval), `realtime.py` (demo), `visualize_results.py`.
- Outputs stored under `exp_result/` per-run for reproducibility.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download and prepare datasets (example for ASVspoof2019 LA):

```bash
python download_dataset.py
```

3. Train a model with a config from `config/`:

```bash
python main.py --config ./config/SimpleCNN.conf
```

Evaluate a trained model:

```bash
python main.py --eval --config ./config/SimpleCNN.conf
```

## Common Configs / Models

- `config/LCNN.conf` / `config/LCNN_Large.conf` — LCNN models (spectrogram input)
- `config/SEResNet.conf` — SEResNet (spectrogram input)
- `config/EfficientNetB2.conf` / `config/EfficientNetB2_Attention.conf` — EfficientNet-B2 variants
- `config/RawNet3.conf` — RawNet3 (raw waveform)

## Datasets (flag `--dataset`)

- `1`: ASVspoof2019 (LA / PA / DF tracks)
- `2`: Fake-or-Real (2s clips)
- `3`: SceneFake

## Feature Types (`--feature_type`)

- `0` — Raw waveform
- `1` — Mel-spectrogram (128 bins)
- `2` — LFCC
- `3` — MFCC
- `4` — CQT

## Useful CLI flags

- `--config`: Path to JSON config file in `config/`.
- `--dataset`: Dataset id (1/2/3).
- `--feature_type`: Feature id (0–4).
- `--epochs`, `--batch_size`: Override config values.
- `--random_noise`: Enable data augmentation (RIR, MUSAN-style noise, pitch shift, time stretch, SpecAugment).
- `--weight_avg`: Enable SWA weight averaging.
- `--eval_best`: Evaluate on test set when a new best model is found.
- `--eval_model_weights`: Path to specific weights for evaluation.
- `--feature_analysis`: Generate feature visualizations for a sample audio file.
- `--cpu`: Force CPU mode.

Example full command:

```bash
python main.py --config config/LCNN.conf --dataset 2 --feature_type 1 --epochs 20 --batch_size 32 --random_noise --weight_avg --eval_best
```

## Output structure (per run)

After training, each experiment folder under `exp_result/` contains:

```
exp_result/
└── <dataset>_<track>_<model>_<flags>_ep<epochs>_bs<batch>_feat<feature>/
    ├── config.conf              # Copy of the used config
    ├── weights/                 # Model checkpoints (best.pth, swa.pth)
    ├── metrics/                 # epoch_metrics.json, final_summary.json
    ├── metric_log.txt
    ├── evaluation_results.txt
    └── events.out.*            # TensorBoard logs
```

## Metrics

- EER — Equal Error Rate (primary metric for spoof detection)
- Accuracy — classification accuracy
- t-DCF — Tandem Detection Cost Function (ASVspoof evaluation)

## Realtime demo

Run `realtime.py` for minimal microphone/file-based inference. Check the top of that file for usage details and example flags.

## Adding models

To add a new model:

1. Implement your model class named `Model` inside `models/` (e.g., `models/MyModel.py`).
2. Create a config file in `config/` referencing the model filename in `model_config.architecture`.
3. Run `python main.py --config ./config/YourConfig.conf`.

## Tips & Common Issues

- Out of memory: reduce `--batch_size` (e.g., 16 or 8).
- Slow training: enable AMP / mixed-precision (configured in JSON), or use `--feature_type` spectrograms for smaller models.
- Quick tests: use `--data_subset 0.1` to train on 10% of data.

## Utilities

- `download_dataset.py` — dataset download/preparation helper
- `visualize_results.py` — generate comparison plots across experiment metrics
- `run_experiments.sh` — example orchestration script

## License

This repository includes an MIT-style license. See the `LICENSE` file for full terms.

## Acknowledgements & References

## Acknowledgements & References

This project builds on ASVspoof baselines and open-source implementations (t-DCF code, RawNet baselines). See project references and the original README/notebook for full citations.
