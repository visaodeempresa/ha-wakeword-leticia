#!/usr/bin/env python3
"""
Train custom wake word "Letícia" for openWakeWord / Home Assistant.

Usage:
    python train_leticia.py --all          # Run everything
    python train_leticia.py --setup        # Install deps + download models
    python train_leticia.py --download     # Download training data
    python train_leticia.py --generate     # Generate clips with Piper pt_BR
    python train_leticia.py --augment      # Augment clips + extract features
    python train_leticia.py --train        # Train model
    python train_leticia.py --convert      # Convert ONNX → TFLite

All data sources verified 2026-03-30 against the official openWakeWord notebook:
  https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
"""

import os
import sys
import argparse
import subprocess
import uuid
import random
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

TARGET_WORD = 'letícia'
MODEL_NAME = 'leticia'
N_POSITIVE = 10000
N_VAL = 2000

PTBR_VOICES = [
    'piper_voices_ptbr/pt_BR-faber-medium.onnx',
    'piper_voices_ptbr/pt_BR-edresson-low.onnx',
]

NEGATIVE_WORDS = [
    'patrícia', 'notícia', 'delícia', 'justiça', 'preguiça',
    'milícia', 'malícia', 'polícia', 'carência', 'urgência',
    'letivo', 'letrada', 'legítima', 'legião', 'elétrica',
    'lícia', 'alícia', 'felícia', 'luciana', 'larissa',
    'olá', 'bom dia', 'boa noite', 'obrigado', 'por favor',
    'ligar', 'desligar', 'acender', 'apagar', 'aumentar',
    'diminuir', 'temperatura', 'música', 'que horas são',
    'televisão', 'computador', 'celular', 'internet', 'cozinha',
]

# Verified URLs (all confirmed via HTTP requests 2026-03-30)
URLS = {
    # Embedding models — release v0.5.1 (confirmed via GitHub API)
    'embedding_model.onnx': 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx',
    'embedding_model.tflite': 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite',
    'melspectrogram.onnx': 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx',
    'melspectrogram.tflite': 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite',

    # LibriTTS v2.0.0 — confirmed via GitHub releases API
    'libritts': 'https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt',

    # Piper voices pt_BR — confirmed via HTTP 302 redirect
    'faber_onnx': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx',
    'faber_json': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json',
    'edresson_onnx': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx',
    'edresson_json': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx.json',

    # ACAV100M features — confirmed file exists on HF (17.3 GB)
    'acav100m': 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy',

    # Validation features — confirmed file exists on HF (185 MB)
    'validation': 'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy',

    # AudioSet tar — confirmed via HF API
    'audioset_tar': 'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar',
}

# HuggingFace datasets (used via datasets library, not wget)
HF_DATASETS = {
    # MIT RIRs — confirmed: 271 WAV files, split='train', format='audiofolder'
    'mit_rirs': {'name': 'davidscripka/MIT_environmental_impulse_responses', 'split': 'train'},
    # FMA — confirmed: name='small', split='train'
    'fma': {'name': 'rudraml/fma', 'config': 'small', 'split': 'train'},
}


def run(cmd, check=True):
    """Run shell command."""
    log.info(f'$ {cmd}')
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0


def wget(url, output, continue_download=False):
    """Download file if it doesn't exist."""
    if os.path.exists(output):
        log.info(f'[SKIP] {output} already exists')
        return
    flags = '-q --show-progress'
    if continue_download:
        flags += ' -c'
    run(f"wget {flags} -O '{output}' '{url}'")


# ============================================================
# Step functions
# ============================================================

def step_setup():
    """Clone repos, install deps, download models."""
    log.info('=' * 60)
    log.info('SETUP: Installing dependencies')
    log.info('=' * 60)

    # Clone repos
    if not os.path.exists('./piper-sample-generator'):
        run('git clone -q https://github.com/dscripka/piper-sample-generator')
    if not os.path.exists('./openwakeword'):
        run('git clone -q https://github.com/dscripka/openWakeWord openwakeword')

    # Install Python packages
    run('pip install -q pathvalidate piper-tts piper-phonemize-cross webrtcvad')
    run("pip install -q 'torch<=2.5' torchvision torchaudio")
    run('pip install -q -e ./openwakeword')
    run('pip install -q mutagen==1.47.0 torchinfo==1.8.0 torchmetrics==1.2.0')
    run('pip install -q speechbrain==0.5.14 audiomentations==0.33.0 torch-audiomentations==0.11.0')
    run('pip install -q acoustics==0.2.6 scipy')
    run('pip install -q onnxruntime onnxsim')
    run('pip install -q onnx2tf tensorflow==2.19.0')
    run('pip install -q onnx==1.19.1 onnx_graphsurgeon')
    run('pip install -q datasets==2.14.6')
    run('cd piper-sample-generator && pip install -q -r requirements.txt')

    # Download embedding models (v0.5.1)
    models_dir = 'openwakeword/openwakeword/resources/models'
    os.makedirs(models_dir, exist_ok=True)
    for fname in ['embedding_model.onnx', 'embedding_model.tflite',
                  'melspectrogram.onnx', 'melspectrogram.tflite']:
        wget(URLS[fname], os.path.join(models_dir, fname))

    # LibriTTS
    os.makedirs('piper-sample-generator/models', exist_ok=True)
    wget(URLS['libritts'], 'piper-sample-generator/models/en_US-libritts_r-medium.pt')

    # Piper pt_BR voices
    os.makedirs('piper_voices_ptbr', exist_ok=True)
    wget(URLS['faber_onnx'], 'piper_voices_ptbr/pt_BR-faber-medium.onnx')
    wget(URLS['faber_json'], 'piper_voices_ptbr/pt_BR-faber-medium.onnx.json')
    wget(URLS['edresson_onnx'], 'piper_voices_ptbr/pt_BR-edresson-low.onnx')
    wget(URLS['edresson_json'], 'piper_voices_ptbr/pt_BR-edresson-low.onnx.json')

    log.info('[OK] Setup complete')


def step_download():
    """Download training datasets."""
    import numpy as np
    import scipy.io.wavfile as wavfile
    import datasets
    from tqdm import tqdm

    log.info('=' * 60)
    log.info('DOWNLOAD: Training data')
    log.info('=' * 60)

    # 1. MIT RIRs
    rir_dir = 'mit_rirs'
    if not os.path.exists(rir_dir) or len(list(Path(rir_dir).glob('*.wav'))) == 0:
        log.info('Downloading MIT Room Impulse Responses...')
        os.makedirs(rir_dir, exist_ok=True)
        ds = datasets.load_dataset(
            HF_DATASETS['mit_rirs']['name'],
            split=HF_DATASETS['mit_rirs']['split'],
            streaming=True
        )
        count = 0
        for row in tqdm(ds, desc='MIT RIRs'):
            name = row['audio']['path'].split('/')[-1]
            audio = np.array(row['audio']['array'])
            wavfile.write(os.path.join(rir_dir, name), 16000, (audio * 32767).astype(np.int16))
            count += 1
        log.info(f'[OK] MIT RIRs: {count} files')
    else:
        log.info(f'[SKIP] MIT RIRs ({len(list(Path(rir_dir).glob("*.wav")))} files)')

    # 2. AudioSet
    as_dir = 'audioset_16k'
    if not os.path.exists(as_dir) or len(os.listdir(as_dir)) == 0:
        log.info('Downloading AudioSet...')
        os.makedirs('audioset', exist_ok=True)
        os.makedirs(as_dir, exist_ok=True)
        tar_path = 'audioset/bal_train09.tar'
        wget(URLS['audioset_tar'], tar_path)
        run(f'cd audioset && tar -xf bal_train09.tar 2>/dev/null', check=False)
        flac_files = list(Path('audioset/audio').glob('**/*.flac'))
        if flac_files:
            log.info(f'Converting {len(flac_files)} FLAC → WAV 16kHz...')
            ds = datasets.Dataset.from_dict({'audio': [str(f) for f in flac_files]})
            ds = ds.cast_column('audio', datasets.Audio(sampling_rate=16000))
            for row in tqdm(ds, desc='AudioSet'):
                name = row['audio']['path'].split('/')[-1].replace('.flac', '.wav')
                audio = np.array(row['audio']['array'])
                wavfile.write(os.path.join(as_dir, name), 16000, (audio * 32767).astype(np.int16))
        log.info(f'[OK] AudioSet: {len(os.listdir(as_dir))} clips')
    else:
        log.info(f'[SKIP] AudioSet ({len(os.listdir(as_dir))} clips)')

    # 3. FMA
    fma_dir = 'fma'
    if not os.path.exists(fma_dir) or len(os.listdir(fma_dir)) == 0:
        log.info('Downloading FMA (1 hour)...')
        os.makedirs(fma_dir, exist_ok=True)
        ds = datasets.load_dataset(
            HF_DATASETS['fma']['name'],
            name=HF_DATASETS['fma']['config'],
            split=HF_DATASETS['fma']['split'],
            streaming=True
        )
        ds_iter = iter(ds.cast_column('audio', datasets.Audio(sampling_rate=16000)))
        n_clips = 120  # 1 hour of 30s clips
        count = 0
        for i in tqdm(range(n_clips), desc='FMA'):
            try:
                row = next(ds_iter)
                name = row['audio']['path'].split('/')[-1].replace('.mp3', '.wav')
                audio = np.array(row['audio']['array'])
                wavfile.write(os.path.join(fma_dir, name), 16000, (audio * 32767).astype(np.int16))
                count += 1
            except StopIteration:
                break
        log.info(f'[OK] FMA: {count} clips')
    else:
        log.info(f'[SKIP] FMA ({len(os.listdir(fma_dir))} clips)')

    # 4. ACAV100M features (17.3 GB)
    acav_file = 'openwakeword_features_ACAV100M_2000_hrs_16bit.npy'
    if not os.path.exists(acav_file):
        log.info('Downloading ACAV100M features (17.3 GB)...')
        wget(URLS['acav100m'], acav_file, continue_download=True)
    else:
        size_gb = os.path.getsize(acav_file) / (1024**3)
        log.info(f'[SKIP] ACAV100M ({size_gb:.1f} GB)')

    # 5. Validation features (185 MB)
    val_file = 'validation_set_features.npy'
    if not os.path.exists(val_file):
        log.info('Downloading validation features (185 MB)...')
        wget(URLS['validation'], val_file)
    else:
        log.info('[SKIP] Validation features')

    log.info('[OK] All data downloaded')


def step_generate():
    """Generate clips with Piper pt_BR voices."""
    log.info('=' * 60)
    log.info('GENERATE: Creating clips with Piper pt_BR')
    log.info('=' * 60)

    length_scales = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25]
    noise_scales = [0.5, 0.6, 0.667, 0.7, 0.8, 0.9, 0.98]
    noise_ws = [0.5, 0.6, 0.7, 0.8, 0.9, 0.98]

    base = f'./my_custom_model/{MODEL_NAME}'
    dirs = {
        'positive_train': f'{base}/positive_train',
        'positive_test': f'{base}/positive_test',
        'negative_train': f'{base}/negative_train',
        'negative_test': f'{base}/negative_test',
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def gen(text, out_dir, n, label):
        existing = len(list(Path(out_dir).glob('*.wav')))
        if existing >= int(n * 0.95):
            log.info(f'[SKIP] {label}: {existing} clips')
            return
        needed = n - existing
        texts = [text] if isinstance(text, str) else text
        count = 0
        t0 = time.time()
        while count < needed:
            for voice in PTBR_VOICES:
                for word in texts:
                    out = os.path.join(out_dir, f'{uuid.uuid4().hex}.wav')
                    try:
                        subprocess.run(
                            ['piper', '--model', voice, '--output_file', out,
                             '--length-scale', str(random.choice(length_scales)),
                             '--noise-scale', str(random.choice(noise_scales)),
                             '--noise-w', str(random.choice(noise_ws))],
                            input=word, capture_output=True, text=True, timeout=30
                        )
                        if os.path.exists(out):
                            count += 1
                    except Exception:
                        pass
                    if count >= needed:
                        break
                if count >= needed:
                    break
            if count % 1000 == 0 and count > 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                log.info(f'{label}: {count}/{needed} ({rate:.1f}/s)')
        total = len(list(Path(out_dir).glob('*.wav')))
        log.info(f'[OK] {label}: {total} clips')

    gen(TARGET_WORD, dirs['positive_train'], N_POSITIVE, 'Positive train')
    gen(TARGET_WORD, dirs['positive_test'], N_VAL, 'Positive val')
    gen(NEGATIVE_WORDS, dirs['negative_train'], N_POSITIVE, 'Negative train')
    gen(NEGATIVE_WORDS, dirs['negative_test'], N_VAL, 'Negative val')

    log.info('[OK] Clip generation complete')


def step_augment():
    """Augment clips and extract features using train.py."""
    import yaml

    log.info('=' * 60)
    log.info('AUGMENT: Augmenting clips + extracting features')
    log.info('=' * 60)

    rir_path = os.path.abspath('mit_rirs') if os.path.exists('mit_rirs') else os.path.abspath('piper-sample-generator/impulses')

    config = {
        'model_name': MODEL_NAME,
        'target_phrase': [TARGET_WORD],
        'custom_negative_phrases': NEGATIVE_WORDS[:9],
        'n_samples': N_POSITIVE,
        'n_samples_val': N_VAL,
        'tts_batch_size': 50,
        'augmentation_batch_size': 16,
        'piper_sample_generator_path': os.path.abspath('./piper-sample-generator'),
        'output_dir': os.path.abspath('./my_custom_model'),
        'rir_paths': [rir_path],
        'background_paths': [os.path.abspath('./audioset_16k'), os.path.abspath('./fma')],
        'background_paths_duplication_rate': [1, 1],
        'augmentation_rounds': 1,
        'false_positive_validation_data_path': os.path.abspath('./validation_set_features.npy'),
        'feature_data_files': {
            'ACAV100M_sample': os.path.abspath('./openwakeword_features_ACAV100M_2000_hrs_16bit.npy')
        },
        'batch_n_per_class': {'ACAV100M_sample': 1024, 'adversarial_negative': 50, 'positive': 50},
        'model_type': 'dnn',
        'layer_size': 32,
        'steps': 50000,
        'max_negative_weight': 1500,
        'target_false_positives_per_hour': 0.2,
    }

    with open('my_model.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    run(f'{sys.executable} openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips')
    log.info('[OK] Augmentation complete')


def step_train():
    """Train the model."""
    log.info('=' * 60)
    log.info('TRAIN: Training model')
    log.info('=' * 60)

    run(f'{sys.executable} openwakeword/openwakeword/train.py '
        f'--training_config my_model.yaml --train_model --convert_to_tflite',
        check=False)

    # Fallback TFLite conversion
    onnx_path = f'my_custom_model/{MODEL_NAME}.onnx'
    tflite_path = f'my_custom_model/{MODEL_NAME}.tflite'
    if os.path.exists(onnx_path) and not os.path.exists(tflite_path):
        log.info('Fallback: converting ONNX → TFLite with onnx2tf...')
        run(f'onnx2tf -i {onnx_path} -o my_custom_model/tf_model -oiqt', check=False)
        import glob, shutil
        tflite_files = glob.glob('my_custom_model/tf_model/**/*.tflite', recursive=True)
        if tflite_files:
            shutil.copy2(tflite_files[0], tflite_path)

    # Report
    for path, label in [(onnx_path, 'ONNX'), (tflite_path, 'TFLite')]:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            log.info(f'✅ {label}: {path} ({size:.1f} KB)')
        else:
            log.warning(f'❌ {label}: not found')

    log.info('[OK] Training complete')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Letícia wake word')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--setup', action='store_true', help='Install deps + download models')
    parser.add_argument('--download', action='store_true', help='Download training data')
    parser.add_argument('--generate', action='store_true', help='Generate clips with Piper')
    parser.add_argument('--augment', action='store_true', help='Augment clips + extract features')
    parser.add_argument('--train', action='store_true', help='Train model + convert')
    args = parser.parse_args()

    if args.all:
        step_setup()
        step_download()
        step_generate()
        step_augment()
        step_train()
    else:
        if args.setup: step_setup()
        if args.download: step_download()
        if args.generate: step_generate()
        if args.augment: step_augment()
        if args.train: step_train()

    if not any(vars(args).values()):
        parser.print_help()
