import os
import random
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

SRC_ROOT = "processed/human"
DST_ROOT = "processed/human_studio_ai_like"

SAMPLE_RATE = 16000
MAX_PER_LANG = 2000   # SAFE, enough

os.makedirs(DST_ROOT, exist_ok=True)

def make_ai_like(wav, sr):
    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample
    if sr != SAMPLE_RATE:
        wav = F.resample(wav, sr, SAMPLE_RATE)

    # light denoise (high-pass)
    wav = F.highpass_biquad(wav, SAMPLE_RATE, 80)

    # compression
    wav = torch.tanh(wav * 2.5)

    # flatten dynamics
    rms = wav.pow(2).mean().sqrt()
    wav = wav / (rms + 1e-6)

    # normalize
    wav = wav / (wav.abs().max() + 1e-9)

    return wav

for lang in os.listdir(SRC_ROOT):
    src_lang = os.path.join(SRC_ROOT, lang)
    dst_lang = os.path.join(DST_ROOT, lang)
    os.makedirs(dst_lang, exist_ok=True)

    files = [f for f in os.listdir(src_lang) if f.endswith(".wav")]
    random.shuffle(files)
    files = files[:MAX_PER_LANG]

    print(f"{lang}: generating {len(files)} AI-like humans")

    for i, f in enumerate(files):
        src = os.path.join(src_lang, f)
        dst = os.path.join(dst_lang, f)

        wav, sr = torchaudio.load(src)
        wav = make_ai_like(wav, sr)
        torchaudio.save(dst, wav, SAMPLE_RATE)

print(" human_studio_ai_like built")