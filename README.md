# Transformer MSS 

Implementación de un sistema de separación de fuentes musicales (Music Source Separation) utilizando Transformers y espectrogramas de Mel. Todavía se encuentra en fase de desarrollo
Este proyecto esta pensado para equipos de baja capacidad de cómputo, todo el entrenamiento se está llevando a cabo en una sola RTX 2070.

##  Descripción

De primeras el dataset es dividido en segmentos de 30 segundos y convertido a espectrogramas de Mel. Esto se hace para reducir enormemente la demanda computacional del modelo.

Los espectrogramas son tratados como imágenes, donde se dividen en porciones. A cada porción se le asigna un vector de embeddings y se procede al entrenamiento, comparando entre la canción completa y el instrumento que queremos aislar. Una vez generado un espectrograma de Mel, se procede a convertirlo de nuevo a señal de audio mediante el vocoder de BigVGAN.

En el repositiorio existe un rar con las diferentes pruebas y resultados obtenidos, algunos consiguiendo ya un aprendizaje considerable cuando se hace overfit sobre una parte pequeña del dataset (10 canciones)

Estoy trabajando activamente para refinar el aprendizaje, que consiga generalizar correctamente y probar a entrenarlo sobre el dataset MSUDB18 entero


##  Estructura general

- `Transformer Train`: Script de entrenamiento.
- `Transformer Inference`: Script de inferencia.
- `VocoderTest`: Pruebas de reconstrucción de audio usando BigVGAN.
- `BigVGAN/`: Vocoder utilizado para reconstruir audio desde Mel Spectrograms.
- `musdb/` y `musdb_wav/`: Datasets externos para entrenamiento y evaluación (ignorados en el repo).

##  Requisitos

- Python 3.10+
- PyTorch
- torchaudio
- librosa
- numpy
- tqdm
- matplotlib
- [BigVGAN](https://github.com/juheon95/BigVGAN)
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html)
