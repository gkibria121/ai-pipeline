import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import streamlit as st


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--eval_model_weights", required=False, type=str, default=None)
    # parse_known_args because Streamlit injects its own args
    args, _ = parser.parse_known_args()
    return args


@st.cache_resource
def load_model(config_path: str, weights_path: str or None, device: torch.device):
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config = config.get("model_config", {})
    arch = model_config.get("architecture")
    if arch is None:
        raise RuntimeError("model_config.architecture missing in config file")

    module = import_module(f"models.{arch}")
    model_variant = model_config.get("model_variant", None)
    if model_variant == "attention":
        _model = getattr(module, "ModelWithAttention")
    elif model_variant == "large":
        _model = getattr(module, "ModelLarge")
    else:
        _model = getattr(module, "Model")

    model = _model(model_config).to(device)

    # If explicit weights_path not provided, try config's model_path
    cfg_weights = config.get("model_path", None)
    final_weights = weights_path if weights_path else cfg_weights
    if final_weights:
        p = Path(final_weights)
        if p.exists():
            st.info(f"Loading weights from: {final_weights}")
            state = torch.load(str(p), map_location=device)
            try:
                model.load_state_dict(state)
            except Exception:
                # try loading nested dicts
                if isinstance(state, dict) and "state_dict" in state:
                    model.load_state_dict(state["state_dict"])
                elif isinstance(state, dict) and "model_state_dict" in state:
                    model.load_state_dict(state["model_state_dict"])
                else:
                    model.load_state_dict(state)
        else:
            st.warning(f"Weights file not found: {final_weights}")
    else:
        st.info("No weights provided; using randomly initialized model (from architecture) ")

    model.eval()
    return model, config


def preprocess_audio(path: str, feature_type: int = 0, sample_rate: int = 16000):
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    # Ensure at least 2 seconds (common for Fake-or-Real / ASVspoof clips)
    target_seconds = 2
    target_len = sample_rate * target_seconds
    if y.shape[0] < target_len:
        pad = target_len - y.shape[0]
        y = np.concatenate([y, np.zeros(pad, dtype=y.dtype)])
    elif y.shape[0] > target_len:
        y = y[:target_len]
    if feature_type == 0:
        # raw waveform -> (1,1,L)
        x = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    elif feature_type == 1:
        # log-mel spectrogram (n_mels chosen reasonably)
        mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
        logmel = librosa.power_to_db(mel)
        x = torch.from_numpy(logmel).float().unsqueeze(0).unsqueeze(0)
    else:
        # fallback to raw
        x = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    return x


def infer(model, tensor: torch.Tensor, device: torch.device):
    tensor = tensor.to(device)
    # Use inference_mode for best performance
    with torch.inference_mode():
        out = model(tensor)
        # many models return (feat, logits)
        if isinstance(out, tuple) or isinstance(out, list):
            logits = out[-1]
        else:
            logits = out

        # logits shape: (B, C) or (B,)
        if logits.dim() == 1 or (logits.dim() > 1 and logits.size(1) == 1):
            # single output -> treat as score for real class
            scores = logits.view(-1).cpu().numpy()
            probs_real = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_real = probs[:, 1]

    return probs_real


def main():
    args = parse_args()

    # Streamlit UI
    st.title("Audio Fake/Real Classifier")
    st.markdown("Upload one or more audio files and click `Classify`.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.markdown(f"**Device:** {device}")

    config_path = args.config
    weights_path = args.eval_model_weights if args.eval_model_weights else None

    # Load model
    try:
        model, config = load_model(config_path, weights_path, device)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        # Attempt to stop Streamlit execution when running via `streamlit run`.
        try:
            st.stop()
        except Exception:
            pass
        # Ensure script exits when run as plain python
        sys.exit(1)

    feature_type = config.get("feature_type", 0)
    sample_rate = config.get("sample_rate", 16000)

    debug = st.sidebar.checkbox("Debug: show tensor/logit info", value=False)
    uploaded = st.file_uploader("Choose audio files", type=["wav", "flac", "mp3"], accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            # show audio player
            st.subheader(up.name)
            try:
                data = up.read()
                # Display audio player from bytes
                try:
                    st.audio(data)
                except Exception:
                    # Some streamlit versions attempt to validate by id; ignore here
                    pass
                # write to a temp file for librosa
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(up.name).suffix) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name

                x = preprocess_audio(tmp_path, feature_type=feature_type, sample_rate=sample_rate)
                probs = infer(model, x, device)
                prob = float(probs[0])
                label = "Real" if prob >= 0.5 else "Fake"
                st.write(f"**Prediction:** {label} — **Score (real):** {prob:.4f}")
                if debug:
                    # Run a detailed forward to show logits for debugging
                    with torch.inference_mode():
                        t = x.to(device)
                        out = model(t)
                        logits = out[-1] if isinstance(out, (tuple, list)) else out
                        st.write("- Tensor shape:", list(x.shape))
                        try:
                            st.write("- Logits shape:", list(logits.shape))
                            l = logits.cpu().numpy()
                            st.write("- Logits (first 8 values):", l.flatten()[:8].tolist())
                            if l.ndim > 1:
                                # show softmaxed probabilities for first token
                                probs_all = F.softmax(torch.tensor(l), dim=1).numpy()
                                st.write("- Softmax (first row):", probs_all[0].tolist())
                        except Exception as e:
                            st.write("- Failed to display logits:", str(e))
            except Exception as e:
                st.error(f"Failed to process {up.name}: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Run with: `streamlit run app.py -- --config path/to/config.conf --eval_model_weights path/to/weights.pth`")


if __name__ == "__main__":
    main()
