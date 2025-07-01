import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa

from meldataset import get_mel_spectrogram
import bigvgan
import os

print("Ruta absoluta:", os.path.abspath("vocals.wav"))
print("¿Existe el archivo?", os.path.exists("vocals.wav"))

# Configuraciones
TARGET_AUDIO_PATH = 'BigVGANgenerated.wav'  # <-- Ruta del archivo de audio
device = "cuda" if torch.cuda.is_available() else "cpu"

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
plt.title("Mel Spectrogram compatible con BigVGAN")
plt.xlabel("Frames")
plt.ylabel("Mel bands")
plt.colorbar()
plt.tight_layout()
plt.show()

# Reconstrucción de audio con BigVGAN
with torch.inference_mode():
    wav_gen = model(mel)  # [1, 1, T]
wav_gen = wav_gen.squeeze(0).cpu()  # [1, T]

# Guardar resultado
torchaudio.save("reconstructed_from_target.wav", wav_gen, model.h.sampling_rate)
print("✅ Audio reconstruido guardado como 'reconstructed_from_target.wav'")
