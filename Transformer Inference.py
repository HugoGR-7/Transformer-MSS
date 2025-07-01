from datasets import load_dataset
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import SpeechT5HifiGan
import math
import librosa
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog

import bigvgan
from meldataset import get_mel_spectrogram
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = load_dataset("danjacobellis/musdb", split="test")
print(dataset)

modelVocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
modelVocoder.remove_weight_norm()
modelVocoder = modelVocoder.eval().to(device)  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # pares
        pe[:, 1::2] = torch.cos(position * div_term)  # impares
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class SimpleAudioTransformer(nn.Module):
    def __init__(self, n_mels=100, dim=256, nhead=8, num_layers=16, dropout=0.2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_proj = nn.Linear(n_mels, dim)  # [batch, seq, n_mels] -> [batch, seq, dim]
        self.pos_encoder = PositionalEncoding(dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True  # muy importante si usas [batch, seq, dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(dim, n_mels)

    def forward(self, x):
        # x: [batch, seq, n_mels]
        x = self.input_proj(x)       # [batch, seq, dim]
        x = self.pos_encoder(x)      # aplica codificación posicional
        x = self.transformer(x)      # [batch, seq, dim]
        x = self.output_proj(x)      # [batch, seq, n_mels]
        return x

class EmbeddingAudioTransformer(nn.Module):
    def __init__(self, n_mels=100, patch_width = 10, dim=256, nhead=4, num_layers=8, dropout=0.2):
        super().__init__()
        self.patch_width = patch_width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_dim = n_mels*patch_width
        self.dim = dim
        self.n_mels = n_mels

        #Asignarle a cada patch su embedding
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        #self.pos_encoder = PositionalEncoding(dim, dropout=dropout)
        max_patches = 281
        self.position_embedding = nn.Parameter(torch.zeros(max_patches, self.dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True  # muy importante si usas [batch, seq, dim]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.patch_unembed = nn.Linear(dim, self.patch_dim)

    def forward(self, mel_spec):
        #print(mel_spec.shape)
        batch_size, n_mels, time_dim = mel_spec.shape
        # Crear patches
        num_patches = time_dim // self.patch_width
        total_needed = num_patches * self.patch_width
        pad_amount = time_dim - total_needed

        # Ahora creamos los patches
        patches = mel_spec.unfold(dimension=2, size=self.patch_width, step=self.patch_width)
        # [B, n_mels, num_patches, patch_width]

        # Reordenamos para obtener la forma final
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, num_patches, n_mels * self.patch_width)

        #print(f"Final: {patches.shape}")  # [B, num_patches, n_mels * patch_width]
        # Embeddings de patches
        embeddings = self.patch_embed(patches)  # [B, num_patches, embedding_dim]
        """
        # Embeddings posicionales
        if self.position_embedding is None or self.position_embedding.size(0) != num_patches:
            self.position_embedding = nn.Parameter(torch.zeros(num_patches, self.dim).to(mel_spec.device))
        embeddings = embeddings + self.position_embedding  # [B, num_patches, embedding_dim]
        """
        position_embedding = self.position_embedding[:num_patches, :].to(mel_spec.device)  # Corta al tamaño actual
        embeddings = embeddings + position_embedding

        # Transformer
        transformed = self.transformer(embeddings)  # [B, num_patches, embedding_dim]

        # Reconstruir patches
        patches_out = self.patch_unembed(transformed)  # [B, num_patches, patch_dim]
        patches_out = patches_out.view(batch_size, num_patches, self.n_mels, self.patch_width)  # [B, num_patches, n_mels, patch_width]
        mel_out = patches_out.permute(0, 2, 1, 3).reshape(batch_size, self.n_mels, num_patches * self.patch_width)
        if pad_amount > 0:
            mel_out = F.pad(mel_out, (0, pad_amount), mode='constant', value=0.0)
            #print(mel_out.shape)
        return mel_out


def waveform_to_mel(waveform):

    tensor = torch.tensor(waveform).float()
    if tensor.ndim == 2:  # Estéreo a mono
        print(f"[INFO] Convertido de estéreo a mono: shape antes = {tensor.shape}")
        tensor = tensor.mean(dim=0)  # (channels, time) -> (time,)
    print(tensor.shape)
    #tensor_down = waveform_downsample(tensor.numpy(),24000)
    #tensor_down = torch.from_numpy(tensor_down)
    print(f"[INFO] Forma del waveform antes del MelSpectrogram: {tensor.shape}")
    mel_spec = get_mel_spectrogram(tensor.unsqueeze(0), modelVocoder.h).to(device)  # [1, n_mels, T]
    
    print(f"[INFO] Forma del mel_spec antes del log: {mel_spec.shape}")
    mel_spec = mel_spec.squeeze(0)  # [n_mels, T]
    print(f"[INFO] Forma del mel_spec final (sin batch): {mel_spec.shape}")

    return mel_spec

def infer_and_reconstruct(sample):
    checkpoint_path = seleccionar_archivo()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        x = waveform_to_mel(sample["mixture"]["array"]).T.unsqueeze(0).to(model.device)
        x = x.permute(0, 2, 1)  # Ahora será [B, n_mels, time]
        print(f"Forma de x: {x.shape}")
        pred_mel = model(x).squeeze(0).T.cpu()
        print(f"Forma de pred_mel: {pred_mel.shape}")
        mel_to_audio(pred_mel, save_path="BigVGANgenerated10000.wav")

def mel_to_audio(mel_tensor: torch.Tensor, save_path: str = "generated.wav"):
    """
    Convierte un mel spectrogram (con forma [1, C_mel, T]) a audio WAV usando BigVGAN.
    """
    mel_tensor = mel_tensor.T.unsqueeze(0)  # Añadimos el batch size que hemos quitado durante el entrenamiento [1, C_mel, T]
    mel_tensor = mel_tensor.to(device)
    with torch.inference_mode():
        wav_gen = modelVocoder(mel_tensor)
    wav_gen = wav_gen.squeeze(0).cpu()  # [1, T] -> [T]
    torchaudio.save(save_path, wav_gen, modelVocoder.h.sampling_rate)
    print(f"Audio generado guardado en: {save_path}")

def seleccionar_archivo(extension=".pth"):
    """Abre un diálogo para seleccionar un archivo .pth y devuelve la ruta."""
    root = Tk()
    root.withdraw()  # Oculta la ventana principal de Tk
    ruta = filedialog.askopenfilename(
        title="Selecciona un checkpoint",
        filetypes=[("Archivos .pth", f"*{extension}")]
    )
    return ruta

def waveform_downsample(waveform, target_sr):
    sample_rate = 44100 
    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
    return waveform

model = EmbeddingAudioTransformer().to(device)
SEGMENT_DURATION = 30
SAMPLE_RATE = 24000  # o 24000, según corresponda

# Suponiendo que dataset[0]["mixture"]["array"] existe
mix = torch.tensor(dataset[0]["mixture"]["array"])
print(mix.shape)
vocals = torch.tensor(dataset[0]["vocals"]["array"])

mix_down = waveform_downsample(mix.numpy(), 24000)
mix_down = torch.from_numpy(mix_down)
sf.write("mix_sample.wav", mix_down.T, 24000)
vocals_down = waveform_downsample(vocals.numpy(), 24000)
vocals_down = torch.from_numpy(vocals_down)
print(mix_down.shape)
# Limita la duración
max_samples = SEGMENT_DURATION * SAMPLE_RATE
mix_down = mix_down[:, :max_samples]
vocals_down = vocals_down[:, :max_samples]
print(mix_down.shape)


# Ahora llamas a tu inferencia
infer_and_reconstruct({
    "mixture": {"array": mix_down},
    "vocals": {"array": vocals_down}
})