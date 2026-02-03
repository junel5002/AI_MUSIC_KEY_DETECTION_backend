import librosa
import tempfile
from pydub import AudioSegment
import os

def load_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        tmp.write(file.file.read())
        webm_path = tmp.name

    wav_path = webm_path.replace(".webm", ".wav")

    # Convert webm → wav
    audio = AudioSegment.from_file(webm_path)
    audio.export(wav_path, format="wav")

    y, sr = librosa.load(wav_path, sr=22050, mono=True)

    os.remove(webm_path)
    os.remove(wav_path)

    return y, sr
