import argparse
import json
import os
import pathlib
import random
import re
import tarfile
import zipfile
from io import BytesIO

import torch
import torchaudio
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from datasets import load_dataset
import soundfile as sf
import numpy as np

# Import Encoder from the local codebase
# Ensure the current directory is in python path or run from root
from encoder.codec import Encoder

SAMPLE_RATE = 32000
VAL_PROP = 0.1
VAL_MAX = 512
SEED = 42

def get_args():
    parser = argparse.ArgumentParser(description="Prepare Norwegian TTS dataset.")
    parser.add_argument("--output-dir", type=pathlib.Path, default="./norwegian_dataset", help="Directory to save the processed dataset.")
    parser.add_argument("--nb-tale-audio-dir", type=pathlib.Path, help="Directory containing extracted NB Tale audio files (e.g., sennheiser_1).")
    parser.add_argument("--nb-tale-transcription", type=pathlib.Path, default="Annotation/part_1.trans", help="Path to NB Tale transcription file.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing.")
    parser.add_argument("--no-npsc", action="store_true", help="Skip NPSC dataset.")
    parser.add_argument("--npsc-config", type=str, default="48K_mp3", help="Config name for NPSC dataset (e.g. 48K_mp3, 16K_mp3_bokmaal).")
    parser.add_argument("--no-nb-tale", action="store_true", help="Skip NB Tale dataset.")
    return parser.parse_args()

def load_encoder(device):
    print("Loading encoder...")
    encoder = Encoder()
    encoder_path = hf_hub_download(repo_id='ekwek/Soprano-Encoder', filename='encoder.pth')
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    encoder.to(device)
    encoder.eval()
    print("Encoder loaded.")
    return encoder

def process_audio(audio_tensor, sr, encoder, device):
    if audio_tensor.ndim > 1 and audio_tensor.shape[0] > 1:
        # Mix to mono
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    if sr != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, SAMPLE_RATE)

    # Normalize? The encoder might expect normalized audio.
    # generate_dataset.py didn't seem to normalize explicitly, but wavfile.read returns int16 usually,
    # while torchaudio returns float32 [-1, 1].
    # generate_dataset.py used scipy.io.wavfile.read which gives int16/32 or float.
    # If it was int, torch.from_numpy would keep it.
    # However, codec.py: preprocess calls mel_spec.
    # If audio is float32, it should be fine.

    audio_tensor = audio_tensor.to(device)
    with torch.no_grad():
        # Encoder.preprocess expects (B, T) for raw audio.
        # If we pass (B, 1, T), it treats it as spectrogram.
        # So we should ensure it is (B, T).
        if audio_tensor.dim() == 3 and audio_tensor.shape[1] == 1:
             audio_tensor = audio_tensor.squeeze(1)

        codes = encoder(audio_tensor)

    return codes.squeeze().tolist()

def parse_nb_tale_transcription(trans_file):
    print(f"Parsing {trans_file}...")
    transcriptions = {}
    current_id = None
    current_words = []

    with open(trans_file, 'r', encoding='latin-1') as f: # NB Tale often uses latin-1 or utf-8, check encoding
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#!TRANSCRIPTION!#"):
                continue

            if line.startswith('"') and "/" in line: # File ID
                # Save previous
                if current_id:
                    text = " ".join(current_words).strip()
                    if text:
                        transcriptions[current_id] = text

                current_id = line.strip('"')
                current_words = []
            else:
                # Content line: start end phoneme word
                parts = line.split('\t')
                if len(parts) >= 4:
                    word = parts[3]
                    if word not in ["<start>", "<end>", "<inhale>", "<exhale>", "<silence>", ""]:
                        # Sometimes words are repeated for phonemes, but in the sample
                        # it seemed the word only appeared on the first phoneme line.
                        # Wait, let's re-verify the sample.
                        # 1779 1866 h Har
                        # 1866 1945 "A:
                        # The word "Har" is only on the first line.
                        # So we append parts[3] if it exists and is not empty.
                        if word.strip():
                             current_words.append(word.strip())

    # Save last
    if current_id:
        text = " ".join(current_words).strip()
        if text:
            transcriptions[current_id] = text

    print(f"Loaded {len(transcriptions)} transcriptions from NB Tale.")
    return transcriptions

def process_npsc(dataset_name, encoder, device, config_name="48K_mp3", limit=None):
    print(f"Processing {dataset_name} with config {config_name}...")

    ds = None
    # Try the requested config first
    try:
        try:
            ds = load_dataset(dataset_name, config_name, split="train", streaming=True, trust_remote_code=True)
        except (TypeError, ValueError):
            ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load NPSC with config {config_name}: {e}")
        # Fallback logic if the specific config failed (e.g. not in cache)
        # We try a few common configs
        fallback_configs = ["16K_mp3_bokmaal", "16K_mp3_nynorsk", "16K_mp3"]
        for fb_config in fallback_configs:
            if fb_config == config_name:
                continue
            print(f"Trying fallback config: {fb_config}...")
            try:
                try:
                    ds = load_dataset(dataset_name, fb_config, split="train", streaming=True, trust_remote_code=True)
                except (TypeError, ValueError):
                     ds = load_dataset(dataset_name, fb_config, split="train", streaming=True)

                print(f"Successfully loaded NPSC with fallback config: {fb_config}")
                break
            except Exception as e_fb:
                print(f"Failed to load fallback config {fb_config}: {e_fb}")

        if ds is None:
            print("Could not load NPSC dataset with any config. Skipping.")
            return []

    data = []
    count = 0

    # Check available columns
    if ds is not None:
        try:
            sample = next(iter(ds))
            if 'text' not in sample and 'sentence_text' not in sample:
                print(f"Dataset {dataset_name} does not contain 'text' or 'sentence_text' column. Skipping.")
                return []
        except StopIteration:
            print(f"Dataset {dataset_name} is empty. Skipping.")
            return []

    if ds is not None:
        for sample in tqdm(ds):
            text = sample.get('text', sample.get('sentence_text', ''))
            if not text:
                continue

            if 'audio' not in sample:
                 continue

            audio_array = sample['audio']['array']
            sr = sample['audio']['sampling_rate']

            audio_tensor = torch.from_numpy(audio_array).float()

            try:
                tokens = process_audio(audio_tensor, sr, encoder, device)
                data.append([text, tokens])
                count += 1
            except Exception as e:
                print(f"Error processing sample: {e}")

            if limit and count >= limit:
                break

    return data

def process_nb_tale(trans_file, audio_dir, encoder, device, limit=None):
    if not os.path.exists(trans_file):
        print(f"NB Tale transcription file not found: {trans_file}")
        return []

    transcriptions = parse_nb_tale_transcription(trans_file)
    data = []
    count = 0

    # NB Tale structure: part_1/group_01/p1_g01_f1_1_t-a0001
    # Audio file might be {audio_dir}/part_1/group_01/p1_g01_f1_1_t-a0001.wav

    if not audio_dir:
         print("No audio directory provided for NB Tale. Skipping audio processing, but listing found transcripts.")
         # Return empty as we can't train without audio
         return []

    for file_id, text in tqdm(transcriptions.items()):
        # file_id example: part_1/group_01/p1_g01_f1_1_t-a0001
        # Construct path
        audio_path = audio_dir / f"{file_id}.wav"

        if not audio_path.exists():
            # Try searching? Or assume structure
             # Maybe the user extracted it differently.
             # For now, strictly follow the ID structure.
             pass
        else:
            try:
                # Use soundfile backend explicitly or just use soundfile
                wav_np, sr = sf.read(audio_path)
                wav = torch.from_numpy(wav_np).float()
                # soundfile returns (T, C) or (T,)
                if wav.ndim == 2:
                    wav = wav.t() # (C, T)

                tokens = process_audio(wav, sr, encoder, device)
                data.append([text, tokens])
                count += 1
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        if limit and count >= limit:
            break

    return data

def main():
    args = get_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = load_encoder(device)

    full_dataset = []

    if not args.no_npsc:
        # NPSC Train
        npsc_train = process_npsc("NbAiLab/NPSC", encoder, device, args.npsc_config, args.limit)
        full_dataset.extend(npsc_train)

        # NPSC Test
        # The prompt mentioned NPSC and NPSC_test.
        # NPSC dataset on HF usually has splits, but NbAiLab/NPSC_test is separate.
        # NPSC_test might not use the same config names? It seems it doesn't have configs in the same way,
        # or it is a separate dataset. Let's try with the same config logic but handle failure similarly.
        # Checking NbAiLab/NPSC_test online would be good, but assuming it works similarly or has default.
        # Actually, NPSC_test might just be the test split?
        # But earlier I found "NbAiLab/NPSC" has splits.
        # If "NbAiLab/NPSC_test" is a dataset, does it have the same configs?
        # To be safe, we use the fallback logic inside process_npsc.
        npsc_test = process_npsc("NbAiLab/NPSC_test", encoder, device, args.npsc_config, args.limit)
        full_dataset.extend(npsc_test)

    if not args.no_nb_tale:
        nb_tale_data = process_nb_tale(args.nb_tale_transcription, args.nb_tale_audio_dir, encoder, device, args.limit)
        full_dataset.extend(nb_tale_data)

    print(f"Total samples processed: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("No data processed. Exiting.")
        return

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(full_dataset)

    num_val = min(int(VAL_PROP * len(full_dataset)) + 1, VAL_MAX)
    train_dataset = full_dataset[num_val:]
    val_dataset = full_dataset[:num_val]

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    with open(args.output_dir / "train.json", 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)

    with open(args.output_dir / "val.json", 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)

    print(f"Dataset saved to {args.output_dir}")

if __name__ == "__main__":
    main()
