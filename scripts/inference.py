### Inference
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