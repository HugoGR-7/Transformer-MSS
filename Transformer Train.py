from datasets import load_dataset
import torchaudio
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import random

import bigvgan
from meldataset import get_mel_spectrogram
from IPython.display import Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Cargamos el dataset
dataset = load_dataset("danjacobellis/musdb", split="test")
print(dataset)

#Vocoder BigVGAN, cargado desde Hugging Face. Para entrenar no es necesario a priori pero necesitamos los parámetros de Mel del Vocoder
modelVocoder = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False)
modelVocoder.remove_weight_norm()
modelVocoder = modelVocoder.eval().to(device)

def waveform_to_mel(waveform):

    tensor = torch.tensor(waveform).float()
    if tensor.ndim == 2:  # Estéreo a mono
        #print(f"[INFO] Convertido de estéreo a mono: shape antes = {tensor.shape}")
        tensor = tensor.mean(dim=0)  # (channels, time) -> (time,)

    #print(f"[INFO] Forma del waveform antes del MelSpectrogram: {tensor.shape}")
    mel_spec = get_mel_spectrogram(tensor.unsqueeze(0), modelVocoder.h).to(device)  # [1, n_mels, T]
    
    #print(f"[INFO] Forma del mel_spec antes del log: {mel_spec.shape}")
    mel_spec = mel_spec.squeeze(0).T  # [n_mels, T]
    #print(f"[INFO] Forma del mel_spec final (sin batch): {mel_spec.shape}")

    return mel_spec

def waveform_downsample(waveform, target_sr):
    sample_rate = 44100 
    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
    return waveform

class MusdbMelDataset(Dataset):
    def __init__(self, root_dir, sample_rate=24000, segment_length=30, data_augmentation=False):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_duration_samples = sample_rate * segment_length
        self.segments = []
        self.data_augmentation = data_augmentation
        for song_folder in os.listdir(root_dir):
            song_path = os.path.join(root_dir, song_folder)
            mix_path = os.path.join(song_path, "mixture.wav")
            vocals_path = os.path.join(song_path, "vocals.wav")

            if os.path.isfile(mix_path) and os.path.isfile(vocals_path):
                mixture, _ = torchaudio.load(mix_path)
                vocals, _ = torchaudio.load(vocals_path)
                if mixture.ndim == 2:
                    mixture = mixture.mean(dim=0)
                if vocals.ndim == 2:
                    vocals = vocals.mean(dim=0)
                #Downsample a 24kHz, que es lo que espera BigVGan
                mixture = waveform_downsample(mixture.numpy(), 24000)
                vocals = waveform_downsample(vocals.numpy(), 24000)
                self.segments.extend(self.split_into_segments(mixture, vocals))

    def split_into_segments(self, mixture, vocals):
        segment_samples = self.segment_length * self.sample_rate

        total_samples = mixture.shape[-1]
        num_segments = (total_samples // segment_samples)*2 - 1
        #print(f"Longitud total: {total_samples}, Tamaño de segmento: {segment_samples}")
        #print(f"Cantidad de segmentos generados: {num_segments}")

        segments = []

        for i in range(num_segments):
            start = i * segment_samples - i*segment_samples//2
            end = start + segment_samples
            #print(f"Segmento {i}: {start} a {end}")
            mixture_segment = mixture[start:end]
            vocals_segment = vocals[start:end]

            if self.data_augmentation:
                mixture_segment, vocals_segment = self.apply_augmentation(mixture_segment, vocals_segment)

            # Convertimos a Mel spectrogram
            mel_x = waveform_to_mel(mixture_segment)
            #plot_mel_spectrogram(mel_x, title=f"Mel Spectrogram - Mixture Segment {i}")      
            mel_y = waveform_to_mel(vocals_segment)
            #plot_mel_spectrogram(mel_y, title=f"Mel Spectrogram - Vocals Segment {i}")
            segments.append((mel_x.T, mel_y.T))  # (time, mel)

        return segments
    
    def apply_augmentation(self, mix, voc):
        if random.random() < 0.5:
            # 50% probabilidad de aplicar ganancia
            gain = random.uniform(0.8, 1.2)
            mix *= gain
            voc *= gain
            max_val = max(abs(mix).max(), abs(voc).max(), 1.0)
            mix = mix / max_val
            voc = voc / max_val

        if random.random() < 0.3:
            # 30% probabilidad de añadir ruido blanco
            noise_level = random.uniform(0.001, 0.01)
            noise = np.random.randn(*mix.shape) * noise_level
            mix += noise
            voc += noise  # mantenemos la relación

        if random.random() < 0.3:
            # 30% probabilidad de shift temporal
            shift = random.randint(1000, 8000)
            mix = np.roll(mix, shift)
            voc = np.roll(voc, shift)

        return mix, voc

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]

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
        
class EmbeddingAudioTransformer(nn.Module):
    def __init__(self, n_mels=100, patch_width = 5, dim=256, nhead=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.patch_width = patch_width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_dim = n_mels*patch_width
        self.dim = dim
        self.n_mels = n_mels

        #Asignarle a cada patch su embedding
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        #self.pos_encoder = PositionalEncoding(dim, dropout=dropout)
        self.position_embedding = None

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

        # Embeddings posicionales
        if self.position_embedding is None or self.position_embedding.size(0) != num_patches:
            self.position_embedding = nn.Parameter(torch.zeros(num_patches, self.dim).to(mel_spec.device))
        embeddings = embeddings + self.position_embedding  # [B, num_patches, embedding_dim]

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

def custom_loss(pred, target):
    return 0.5 * F.mse_loss(pred, target) + 0.5 * F.l1_loss(pred, target)
#Elegimos entre entrenamiento desde cero o checkpoint
Eleccion = int(input("1 = Empezar de 0 \n 2 = Elegir checkpoint"))
if(Eleccion == 1):
    model = EmbeddingAudioTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

else:
    model = EmbeddingAudioTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    checkpoint = torch.load("hola.pth", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    #scheduler


train_dataset = MusdbMelDataset("C:/Users/Hugo/Desktop/HuggingFace/musdb_wav/train", data_augmentation = False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = MusdbMelDataset("C:/Users/Hugo/Desktop/HuggingFace/musdb_wav/test", data_augmentation = False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

best_loss = float("inf")
log_file = "checkpoints/training_log.txt"
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(100000):
    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for x, y in train_loader:
        x, y = x.to(model.device), y.to(model.device)
        pred = model(x)

        loss = custom_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step(loss)

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_batches = 0
        for x_val, y_val in test_loader:
            x_val, y_val = x_val.to(model.device), y_val.to(model.device)
            pred_val = model(x_val)
            loss_val = custom_loss(pred_val, y_val)
            val_loss += loss_val.item()
            val_batches += 1

        avg_val_loss = val_loss / val_batches

    if (epoch > 50000) and (avg_val_loss < best_loss):
        checkpoint_path = f"checkpoints/checkpoint_loss{avg_loss:.4f}_checkpoint_valloss{avg_val_loss:.4f}_epoch{epoch}.pth"
        print(f"Modelo guardado: {checkpoint_path}")
        with open(log_file, "a") as f:
            f.write(f"[EPOCH {epoch}] Model saved. Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n")
        best_loss = avg_val_loss
        torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "val_loss":avg_val_loss
            }, checkpoint_path)
    
    print(f"Epoch {epoch} - Avg. Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")

    log_line = f"[EPOCH {epoch}] Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
    print(log_line)
    with open(log_file, "a") as f:
        f.write(log_line + "\n")


