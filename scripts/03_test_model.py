#!/usr/bin/env python3
"""
Script 03: Testar modelo wake word "Letícia" localmente
-------------------------------------------------------
Testa o modelo .onnx com o microfone do computador antes de implantar no HA.

Requisitos:
    pip install openwakeword pyaudio numpy

Uso:
    python scripts/03_test_model.py --model ./models/leticia.onnx
    python scripts/03_test_model.py --model ./models/leticia.tflite
"""

import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Testar wake word Letícia localmente")
    parser.add_argument("--model", required=True, help="Caminho do modelo (.onnx ou .tflite)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold de detecção (0.0-1.0, padrão: 0.5)")
    parser.add_argument("--chunks", type=int, default=1280, help="Tamanho do chunk de áudio")
    args = parser.parse_args()

    try:
        import openwakeword
        from openwakeword.model import Model
        import pyaudio
        import numpy as np
    except ImportError as e:
        print(f"[ERRO] Dependência faltando: {e}")
        print("  Instale com: pip install openwakeword pyaudio numpy")
        sys.exit(1)

    # Carregar modelo
    print(f"Carregando modelo: {args.model}")
    print(f"Threshold: {args.threshold}")

    oww_model = Model(
        wakeword_models=[args.model],
        inference_framework="tflite" if args.model.endswith(".tflite") else "onnx",
    )

    # Configurar áudio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=args.chunks,
    )

    print("\n" + "=" * 50)
    print("  TESTE DE WAKE WORD 'LETÍCIA'")
    print("  Fale 'Letícia' no microfone...")
    print("  Ctrl+C para sair")
    print("=" * 50 + "\n")

    detections = 0
    start_time = time.time()

    try:
        while True:
            audio_data = stream.read(args.chunks, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Inferência
            prediction = oww_model.predict(audio_array)

            for model_name, score in prediction.items():
                if score > args.threshold:
                    detections += 1
                    elapsed = time.time() - start_time
                    print(
                        f"  >>> DETECTADO! [{detections}] "
                        f"Score: {score:.3f} | "
                        f"Tempo: {elapsed:.1f}s"
                    )

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n{'=' * 50}")
        print(f"  Sessão encerrada")
        print(f"  Duração: {elapsed:.0f}s")
        print(f"  Detecções: {detections}")
        print(f"  Taxa: {detections / (elapsed / 60):.1f} detecções/min")
        print(f"{'=' * 50}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
