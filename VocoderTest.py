import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib
import librosa

from meldataset import get_mel_spectrogram
import bigvgan
import os
from scipy.io import wavfile
import numpy as np

matplotlib.use("TkAgg")
print("Ruta absoluta:", os.path.abspath("vocals.wav"))
print("¿Existe el archivo?", os.path.exists("vocals.wav"))

mix_path = "C:\\Users\\Hugo\\Desktop\\Transformer MSS\\musdb_wav\\test\\Lyndsey Ollard - Catching Up\\vocals.wav"
# Configuraciones
TARGET_AUDIO_PATH = mix_path  # <-- Ruta del archivo de audio
device = "cuda" if torch.cuda.is_available() else "cpu"

#Visualizar wav

sample_rate, data = wavfile.read(mix_path)

# 2. Crear el eje de tiempo en segundos
tiempo = np.linspace(0, len(data) / sample_rate, num=len(data))

# 3. Graficar la forma de onda
plt.figure(figsize=(12, 4))
plt.plot(tiempo, data[:,0])  # Canal izquierdo
plt.title("Cancion")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")


# 4. Guardar la imagen
plt.savefig("forma_onda.png", dpi=300)  # Guardar con buena resolución
plt.show()

# Cargar modelo BigVGAN
model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
model.remove_weight_norm()
model = model.eval().to(device)

# Cargar audio y resamplear si hace falta
#wav, sr = librosa.load(TARGET_AUDIO_PATH, sr=model.h.sampling_rate, mono=True)  # sr = 24000
#wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)  # [1, T]

wav_tensor, sr = torchaudio.load(TARGET_AUDIO_PATH)
if sr != model.h.sampling_rate:
    resampler = torchaudio.transforms.Resample(sr, model.h.sampling_rate)
    wav_tensor = resampler(wav_tensor)

# Convertir a mono si tiene más de un canal
if wav_tensor.shape[0] > 1:
    wav_tensor = torch.mean(wav_tensor, dim=0, keepdim=True)


wav_tensor = wav_tensor.to(device)



# Obtener mel spectrogram con los mismos parámetros que BigVGAN espera
mel = get_mel_spectrogram(wav_tensor, model.h).to(device)  # [1, n_mels, T]

# Visualización del Mel spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(mel.squeeze().cpu(), origin="lower", aspect="auto", cmap="magma")
plt.title("Espectrograma de Mel de una canción completa")
plt.xlabel("Ventanas temporales")
plt.ylabel("Bandas de Mel")
plt.colorbar()
plt.tight_layout()
plt.tight_layout()
plt.savefig("mel_spectrogram.png")
print("✅ Mel spectrogram guardado como mel_spectrogram.png")
plt.show()

# Reconstrucción de audio con BigVGAN
with torch.inference_mode():
    wav_gen = model(mel)  # [1, 1, T]
wav_gen = wav_gen.squeeze(0).cpu()  # [1, T]

# Guardar resultado
torchaudio.save("reconstructed_from_target.wav", wav_gen, model.h.sampling_rate)
print("✅ Audio reconstruido guardado como 'reconstructed_from_target.wav'")
