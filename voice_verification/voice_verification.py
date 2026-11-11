"""
voice_verification.py - shared audio code + verify_voice() used by the CLI app.

Import in the CLI:
    from voice_verification import verify_voice
    result = verify_voice("clip.wav", expected_user="blessing")
"""
import os
import numpy as np
import pandas as pd
import librosa
import joblib

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "voice_model.joblib")


# ---------- loading ----------
def load_audio(path):
    """Load as mono 16 kHz and trim leading/trailing silence."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    return y


# ---------- augmentation ----------
def augment(y, sr=SR, seed=42):
    """Return {name: signal}. Small changes only, so speaker identity is kept."""
    rng = np.random.default_rng(seed)
    noise = lambda amp: y + amp * rng.standard_normal(len(y))
    return {
        "orig":         y,
        "pitch_up":     librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "pitch_down":   librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
        "stretch_fast": librosa.effects.time_stretch(y, rate=1.1),
        "stretch_slow": librosa.effects.time_stretch(y, rate=0.9),
        "noise_low":    noise(0.003),
        "noise_high":   noise(0.01),
    }


# ---------- features ----------
def extract_features(y, sr=SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    f = {}
    for i in range(13):
        f[f"mfcc{i+1}_mean"] = float(mfcc[i].mean())
        f[f"mfcc{i+1}_std"] = float(mfcc[i].std())
    f["spectral_rolloff"] = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())
    f["spectral_centroid"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    f["spectral_bandwidth"] = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    f["zcr"] = float(librosa.feature.zero_crossing_rate(y).mean())
    f["rms_energy"] = float(librosa.feature.rms(y=y).mean())
    f["energy"] = float(np.sum(y ** 2) / len(y))
    return f


# ---------- verification ----------
def verify_voice(path, expected_user=None, threshold=None):
    """
    Returns a dict:
      authorized (bool), predicted_user (str|None), confidence (float), reason (str)

    Authorized only if
      (1) the voice is not classed as 'unknown' (outsider),
      (2) the model's top confidence >= threshold, and
      (3) when expected_user is given (the face-recognised user),
          the voice matches that same person.
    """
    bundle = joblib.load(MODEL_PATH)
    model, cols = bundle["model"], bundle["feature_cols"]
    thr = bundle["threshold"] if threshold is None else threshold

    y = load_audio(path)
    x = pd.DataFrame([extract_features(y)])[cols]
    proba = model.predict_proba(x)[0]
    best = int(np.argmax(proba))
    who, conf = str(model.classes_[best]), float(proba[best])

    if who == "unknown":
        return dict(authorized=False, predicted_user=None, confidence=conf,
                    reason="voice classified as an unauthorized speaker")
    if conf < thr:
        return dict(authorized=False, predicted_user=None, confidence=conf,
                    reason=f"voice not recognised (confidence {conf:.2f} < {thr:.2f})")
    if expected_user and who.lower() != expected_user.lower():
        return dict(authorized=False, predicted_user=who, confidence=conf,
                    reason=f"voice belongs to '{who}', not the face-verified user '{expected_user}'")
    return dict(authorized=True, predicted_user=who, confidence=conf, reason="voice verified")


if __name__ == "__main__":
    import sys
    print(verify_voice(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))