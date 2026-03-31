#!/usr/bin/env python3
"""
Script 01: Setup para treinamento da wake word "Letícia" com vozes pt_BR
------------------------------------------------------------------------
Este script é destinado ao Google Colab. Ele substitui o bloco de setup
do notebook oficial do openWakeWord para usar vozes em português brasileiro.

Uso: Cole este código no primeiro bloco do Colab notebook:
https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb
"""

import subprocess
import os
import sys

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================

TARGET_WORD = "letícia"

# Vozes Piper pt_BR disponíveis (todas single-speaker)
PIPER_VOICES_PTBR = {
    "faber-medium": {
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json",
    },
    "edresson-low": {
        "model": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx",
        "config": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx.json",
    },
}

# Parâmetros de variação para compensar single-speaker
LENGTH_SCALES = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
NOISE_SCALES = [0.5, 0.6, 0.667, 0.7, 0.8]

SAMPLES_PER_VOICE = 500  # Total por voz (variando length/noise)
OUTPUT_DIR = "positive_samples"
NEGATIVE_DIR = "negative_samples"

# ==============================================================================
# INSTALAÇÃO DE DEPENDÊNCIAS
# ==============================================================================


def install_dependencies():
    """Instala pacotes necessários no ambiente Colab."""
    packages = [
        "piper-tts",
        "openwakeword",
        "tensorflow==2.19.0",
        "onnx==1.17.0",
        "onnxruntime==1.18.1",
        "onnx2tf",
        "librosa",
        "soundfile",
    ]
    for pkg in packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-q"], check=True
        )
    print("[OK] Dependências instaladas.")


# ==============================================================================
# DOWNLOAD DE MODELOS
# ==============================================================================


def download_voices():
    """Baixa modelos de voz Piper pt_BR."""
    os.makedirs("voices", exist_ok=True)

    for name, urls in PIPER_VOICES_PTBR.items():
        model_path = f"voices/pt_BR-{name}.onnx"
        config_path = f"voices/pt_BR-{name}.onnx.json"

        if not os.path.exists(model_path):
            print(f"Baixando voz: {name}...")
            subprocess.run(["wget", "-q", "-O", model_path, urls["model"]], check=True)
            subprocess.run(
                ["wget", "-q", "-O", config_path, urls["config"]], check=True
            )
            print(f"  [OK] {name}")
        else:
            print(f"  [SKIP] {name} já existe")


# ==============================================================================
# GERAÇÃO DE AMOSTRAS
# ==============================================================================


def generate_positive_samples():
    """Gera amostras positivas da wake word com variações."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_count = 0
    for voice_name in PIPER_VOICES_PTBR:
        model_path = f"voices/pt_BR-{voice_name}.onnx"

        for ls in LENGTH_SCALES:
            for ns in NOISE_SCALES:
                output_file = f"{OUTPUT_DIR}/leticia_{voice_name}_ls{ls}_ns{ns}_{sample_count:05d}.wav"

                cmd = [
                    "piper",
                    "--model", model_path,
                    "--output_file", output_file,
                    "--length-scale", str(ls),
                    "--noise-scale", str(ns),
                ]

                try:
                    process = subprocess.run(
                        cmd,
                        input=TARGET_WORD,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    sample_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"  [ERRO] {e.stderr}")

                if sample_count >= SAMPLES_PER_VOICE * len(PIPER_VOICES_PTBR):
                    break

    print(f"[OK] {sample_count} amostras positivas geradas em {OUTPUT_DIR}/")
    return sample_count


def generate_negative_samples():
    """Gera amostras negativas (palavras que NÃO são a wake word)."""
    os.makedirs(NEGATIVE_DIR, exist_ok=True)

    # Palavras foneticamente similares (para treinar discriminação)
    # + palavras comuns em português
    negative_words = [
        "patrícia", "notícia", "delícia", "justiça", "preguiça",
        "milícia", "malícia", "polícia", "carência", "urgência",
        "olá", "bom dia", "boa noite", "obrigado", "por favor",
        "televisão", "computador", "celular", "internet", "cozinha",
        "sala", "quarto", "banheiro", "janela", "porta",
        "ligar", "desligar", "acender", "apagar", "aumentar",
        "diminuir", "temperatura", "música", "hora", "tempo",
    ]

    sample_count = 0
    for voice_name in PIPER_VOICES_PTBR:
        model_path = f"voices/pt_BR-{voice_name}.onnx"

        for word in negative_words:
            for ls in [0.9, 1.0, 1.1]:
                output_file = (
                    f"{NEGATIVE_DIR}/neg_{voice_name}_{sample_count:05d}.wav"
                )

                cmd = [
                    "piper",
                    "--model", model_path,
                    "--output_file", output_file,
                    "--length-scale", str(ls),
                ]

                try:
                    subprocess.run(
                        cmd,
                        input=word,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    sample_count += 1
                except subprocess.CalledProcessError:
                    pass

    print(f"[OK] {sample_count} amostras negativas geradas em {NEGATIVE_DIR}/")
    return sample_count


# ==============================================================================
# RESAMPLE PARA 16kHz (requisito do openWakeWord)
# ==============================================================================


def resample_all_to_16khz():
    """Reamostra todos os WAVs para 16kHz mono (requisito do embedding model)."""
    import librosa
    import soundfile as sf

    for directory in [OUTPUT_DIR, NEGATIVE_DIR]:
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                filepath = os.path.join(directory, filename)
                audio, sr = librosa.load(filepath, sr=16000, mono=True)
                sf.write(filepath, audio, 16000)

    print("[OK] Todas as amostras reamostradas para 16kHz mono.")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Wake Word 'Letícia' - Setup de Treinamento")
    print("=" * 60)

    print("\n[1/5] Instalando dependências...")
    install_dependencies()

    print("\n[2/5] Baixando vozes Piper pt_BR...")
    download_voices()

    print("\n[3/5] Gerando amostras positivas...")
    pos_count = generate_positive_samples()

    print("\n[4/5] Gerando amostras negativas...")
    neg_count = generate_negative_samples()

    print("\n[5/5] Reamostando para 16kHz...")
    resample_all_to_16khz()

    print("\n" + "=" * 60)
    print(f"  CONCLUÍDO!")
    print(f"  Positivas: {pos_count} | Negativas: {neg_count}")
    print(f"  Próximo passo: Execute o treinamento do openWakeWord")
    print("=" * 60)
