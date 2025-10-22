# Music Analysis — Extração de Espectrograma e Features

Uma ferramenta simples para analisar uma faixa de áudio (ex.: `Free.mp3`) de forma rápida e previsível. O script principal (`app.py`) lê o áudio, gera um espectrograma de amplitude e extrai features acústicas clássicas, salvando resultados em arquivos prontos para inspeção ou processamento posterior.

## O que será executado (visão geral)
- Carrega o arquivo `Free.mp3` (ou gera um sinal de teste se o arquivo não existir).
- Calcula e salva um espectrograma de amplitude como imagem (`free_spectrogram.png`).
- Extrai diversas features do áudio (ZCR, RMSE, centroid, bandwidth, rolloff, MFCCs, chroma).
- Salva as features como CSV (`free_features.csv`) e JSON (`free_features.json`).

## Saídas geradas
- `free_spectrogram.png` — imagem do espectrograma (dB, escala log).
- `free_features.csv` — uma linha com todas as features extraídas.
- `free_features.json` — mesmas features em formato JSON legível.

## Requisitos
- Python 3.8+
- Bibliotecas:
  - numpy, librosa, matplotlib, pandas
- Instalação rápida (PowerShell):
```powershell
python -m pip install numpy librosa matplotlib pandas
```

## Como executar
1. Coloque `Free.mp3` na pasta do projeto (opcional — se ausente, um sinal de teste será usado).
2. Abra o terminal integrado (PowerShell) na pasta:
```powershell
cd "c:\Users\anton\OneDrive\Área de Trabalho\Music Analysis"
python app.py
```
3. Verifique os arquivos de saída na mesma pasta.

## Boas práticas / dicas
- Ajuste parâmetros (taxa de amostragem, n_mfcc, hop_length) diretamente em `app.py` se precisar de resolução diferente.
- Use os arquivos CSV/JSON para análises estatísticas, visualizações ou como entrada para modelos de machine learning.
- Para análises em lote, adapte o script para iterar sobre vários arquivos na pasta.

Pronto — execute `app.py` para obter um espectrograma visual e um conjunto de features numéricas da sua música.
