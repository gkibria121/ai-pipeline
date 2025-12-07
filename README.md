# ============================================================================

# FILE: README.md

# ============================================================================

````markdown
# AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention

PyTorch implementation of AASIST for audio deepfake detection.

## Overview

AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention) is a state-of-the-art audio deepfake detection system that uses graph attention networks to model spectro-temporal relationships in audio signals.

**Reference Paper:**

> Jung, J. et al. (2022). "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks". In ICASSP 2022.

## Features

- ✅ Graph Attention Networks (GAT) for spectral and temporal modeling
- ✅ Heterogeneous graph attention for cross-domain interactions
- ✅ Sinc-based frontend for raw waveform processing
- ✅ Multiple inference paths with learnable master nodes
- ✅ Support for frequency augmentation during training

## Installation

### Requirements

```bash
pip install -r requirements.txt
```
````

### Dataset

Download the ASVspoof 2019 LA dataset:

1. Visit: https://www.asvspoof.org/index2019.html
2. Download and extract to `./LA/` directory

Expected directory structure:

```
LA/
├── ASVspoof2019_LA_train/
│   └── flac/
├── ASVspoof2019_LA_dev/
│   └── flac/
├── ASVspoof2019_LA_eval/
│   └── flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

## Usage

### Training

```bash
python scripts/train.py --config config/config.json --seed 1234
```

**Training Options:**

- `--config`: Path to configuration file
- `--seed`: Random seed for reproducibility (default: 1234)

### Evaluation

Evaluate on development set:

```bash
python scripts/evaluate.py --config config/config.json --eval_set dev
```

Evaluate on evaluation set:

```bash
python scripts/evaluate.py --config config/config.json --eval_set eval
```

**Evaluation Options:**

- `--config`: Path to configuration file
- `--model_path`: Override model path from config
- `--eval_set`: Choose 'dev' or 'eval' set

### Inference

```python
import torch
from models import AASIST
import soundfile as sf

# Load model
config = {...}  # Your model config
model = AASIST(config)
model.load_state_dict(torch.load('path/to/weights.pth'))
model.eval()

# Load audio
audio, sr = sf.read('audio.flac')
audio_tensor = torch.FloatTensor(audio).unsqueeze(0)

# Inference
with torch.no_grad():
    _, output = model(audio_tensor)
    score = output[0, 1] - output[0, 0]  # bonafide - spoof

print(f"Score: {score.item():.4f}")
print(f"Prediction: {'Bonafide' if score > 0 else 'Spoof'}")
```

## Configuration

The configuration file (`config/config.json`) contains all hyperparameters:

```json
{
  "database_path": "./LA/",
  "model_path": "./models/weights/AASIST.pth",
  "batch_size": 24,
  "num_epochs": 100,
  "model_config": {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
  },
  "optim_config": {
    "optimizer": "adam",
    "base_lr": 0.0001,
    "lr_min": 0.000005,
    "weight_decay": 0.0001,
    "scheduler": "cosine"
  }
}
```

### Key Parameters

**Model Architecture:**

- `nb_samp`: Input audio length in samples (~4 seconds at 16kHz)
- `first_conv`: Sinc filter kernel size
- `filts`: Channel dimensions for residual blocks
- `gat_dims`: Dimensions for GAT layers
- `pool_ratios`: Graph pooling ratios
- `temperatures`: Attention temperature parameters

**Training:**

- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `optimizer`: Optimizer type ('adam' or 'sgd')
- `base_lr`: Initial learning rate
- `scheduler`: Learning rate scheduler ('cosine', 'multistep', 'sgdr')

## Model Architecture

### Overview

```
Raw Audio → SincConv → ResBlocks → GAT-S & GAT-T → HtrgGAT → Output
                                        ↓
                                   Master Nodes
```

### Components

1. **SincConv Frontend**: Learnable bandpass filters for raw waveform
2. **Residual Encoder**: 6 residual blocks for feature extraction
3. **GAT-S**: Graph attention for spectral features
4. **GAT-T**: Graph attention for temporal features
5. **HtrgGAT**: Heterogeneous graph attention for cross-domain modeling
6. **Master Nodes**: Learnable global representations
7. **Dual Path**: Two parallel inference paths for robustness

## Evaluation Metrics

- **EER** (Equal Error Rate): The point where FAR = FRR
- **min t-DCF** (minimum tandem Detection Cost Function): Detection cost in tandem with ASV system

Lower values indicate better performance.

## Project Structure

```
aasist_project/
├── config/
│   └── config.json
├── models/
│   ├── __init__.py
│   ├── aasist.py
│   └── layers.py
├── utils/
│   ├── __init__.py
│   ├── optimizer.py
│   └── metrics.py
├── data_utils/
│   ├── __init__.py
│   └── dataset.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── checkpoints/
│   └── weights/
├── requirements.txt
└── README.md
```

## Results

Expected performance on ASVspoof 2019 LA:

| Metric    | Development | Evaluation |
| --------- | ----------- | ---------- |
| EER       | ~0.83%      | ~0.99%     |
| min t-DCF | ~0.0275     | ~0.0352    |

_Note: Results may vary depending on training settings and random seed._

## Tips for Training

1. **Use GPU**: Training is significantly faster with CUDA
2. **Batch Size**: Adjust based on available GPU memory
3. **Learning Rate**: Start with 0.0001 and use cosine annealing
4. **Augmentation**: Frequency masking helps prevent overfitting
5. **Checkpoints**: Save regularly to prevent data loss

## Troubleshooting

### Common Issues

**Out of Memory:**

```python
# Reduce batch size in config
"batch_size": 16  # or 8
```

**Slow Training:**

```python
# Increase num_workers in DataLoader
train_loader = DataLoader(..., num_workers=8)
```

**Poor Performance:**

- Check data preprocessing
- Verify audio sampling rate (16kHz)
- Ensure proper normalization
- Try different random seeds

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{jung2022aasist,
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6367--6371},
  year={2022},
  organization={IEEE}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub or contact the original authors.

## Acknowledgements

- Original AASIST implementation by NAVER Corp.
- ASVspoof 2019 challenge organizers
- PyTorch team for the deep learning framework

```

```
