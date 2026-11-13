import os, glob
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa, librosa.display
from voice_verification import load_audio, augment, extract_features, SR

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(HERE, "data", "audio")
UNAUTH_DIR = os.path.join(HERE, "data", "audio_unauthorized")
AUG_DIR = os.path.join(HERE, "data", "audio_aug")
PLOT_DIR = os.path.join(HERE, "plots")
os.makedirs(AUG_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)


def parse_name(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    member = stem.split("_")[0]
    phrase = "yesapprove" if "approve" in stem else "confirm" if "confirm" in stem else "other"
    return member, phrase


def plot_clip(y, name, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
    librosa.display.waveshow(y, sr=SR, ax=ax[0])
    ax[0].set(title=f"Waveform - {name}", xlabel="Time (s)", ylabel="Amplitude")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="hz", ax=ax[1])
    ax[1].set(title=f"Spectrogram - {name}")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


files = [(p, False) for p in sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))] + \
        [(p, True) for p in sorted(glob.glob(os.path.join(UNAUTH_DIR, "*.wav")))]

rows = []
for path, is_outsider in files:
    base = os.path.splitext(os.path.basename(path))[0]
    member, phrase = parse_name(path)
    if is_outsider:
        member = "unknown"
    y = load_audio(path)
    plot_clip(y, base, os.path.join(PLOT_DIR, f"{base}.png"))

    for aug_name, y_aug in augment(y).items():
        if aug_name != "orig":
            sf.write(os.path.join(AUG_DIR, f"{base}__{aug_name}.wav"), y_aug, SR)
        row = extract_features(y_aug)
        row.update(member=member, phrase=phrase, source_file=base, augmentation=aug_name)
        rows.append(row)
    print("processed", base, "->", member)

df = pd.DataFrame(rows)
meta = ["member", "phrase", "source_file", "augmentation"]
df = df[meta + [c for c in df.columns if c not in meta]]
df.to_csv(os.path.join(HERE, "audio_features.csv"), index=False)
print(f"saved audio_features.csv  shape={df.shape}  classes={sorted(df.member.unique())}")
