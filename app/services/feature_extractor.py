import librosa
import numpy as np

def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return np.mean(chroma, axis=1)
