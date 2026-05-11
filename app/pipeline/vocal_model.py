import os
import numpy as np
import torch
import torch.nn as nn
import librosa

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "checkpoint.pt")
_SR = 16000
_N_MELS = 128
_HOP = 256
_DURATION = 2.0
_N_FRAMES = int(_SR * _DURATION / _HOP)

_model = None
_X_mean = None
_X_std = None


class VocalPitchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 37),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _load_model():
    global _model, _X_mean, _X_std
    if _model is not None:
        return True
    path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(path):
        return False
    ck = torch.load(path, map_location="cpu")
    _model = VocalPitchCNN()
    _model.load_state_dict(ck["model"])
    _model.eval()
    _X_mean = float(ck["X_mean"])
    _X_std = float(ck["X_std"])
    return True


def correct_pitch(audio_samples: np.ndarray, sr: int, basic_pitch_midi: int, confidence_threshold: float = 0.7) -> int:
    if not _load_model():
        return basic_pitch_midi

    audio = librosa.resample(audio_samples, orig_sr=sr, target_sr=_SR) if sr != _SR else audio_samples
    audio = np.pad(audio, (0, max(0, int(_SR * _DURATION) - len(audio))))[:int(_SR * _DURATION)]

    mel = librosa.feature.melspectrogram(y=audio, sr=_SR, n_mels=_N_MELS, hop_length=_HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = mel_db[:, :_N_FRAMES]
    if mel_db.shape[1] < _N_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, _N_FRAMES - mel_db.shape[1])))

    mel_norm = (mel_db - _X_mean) / (_X_std + 1e-8)
    x = torch.tensor(mel_norm[None, None, :, :])

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)
        cnn_confidence = float(probs.max())
        cnn_midi = int(probs.argmax()) + 48

    if cnn_confidence >= confidence_threshold:
        return cnn_midi
    return basic_pitch_midi
