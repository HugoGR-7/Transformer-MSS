import os
import stempeg
import soundfile as sf
import numpy as np

input_root = "C:/Users/Hugo/Desktop/HuggingFace/musdb"
output_root = "C:/Users/Hugo/Desktop/HuggingFace/musdb_wav"

stem_names = ["mixture", "drums", "bass", "other", "vocals"]

for split in ["train", "test"]:
    input_path = os.path.join(input_root, split)
    output_path = os.path.join(output_root, split)
    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        if not file.endswith(".stem.mp4"):
            continue

        song_path = os.path.join(input_path, file)
        song_name = os.path.splitext(os.path.splitext(file)[0])[0]
        song_output_dir = os.path.join(output_path, song_name)
        os.makedirs(song_output_dir, exist_ok=True)

        try:
            audio, rate = stempeg.read_stems(song_path)
        except Exception as e:
            print(f"Error leyendo {song_path}: {e}")
            continue

        for i, stem_name in enumerate(stem_names):
            data = audio[i]  # ← esto selecciona la pista individual (drums, bass, etc.)
            filepath = os.path.join(song_output_dir, f"{stem_name}.wav")

            # Asegurarse de que tiene forma correcta (samples, channels)
            if data.ndim == 1:
                data = data[:, None]  # convertir mono a (samples, 1)

            # Guardar el stem
            sf.write(filepath, data, rate, format="WAV", subtype="PCM_16")
