"""
train_voice_model.py - trains + evaluates the voiceprint verification model.

Classes = each team member + 'unknown' (outsiders).

Evaluation is leave-one-CLIP-out: all rows (original + augmented) of one
recording are held out together, and we score only the held-out ORIGINAL clip.
This prevents leakage from augmented copies and works with very few recordings.

Run:  python train_voice_model.py
"""
import os
import warnings
warnings.filterwarnings("ignore", message="Unknown solver options")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "audio_features.csv"))
META = ["member", "phrase", "source_file", "augmentation"]
cols = [c for c in df.columns if c not in META]
X, y, groups = df[cols], df["member"], df["source_file"]

models = {
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
    "LogisticRegression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")),
}

results = {}
classes = np.array(sorted(y.unique()))
for name, model in models.items():
    true, pred, probas = [], [], []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        te = [i for i in te if df.augmentation.iloc[i] == "orig"]   # score originals only
        if not te:
            continue
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])
        full = np.zeros((len(te), len(classes)))                      # align if a class is missing
        for j, c in enumerate(model.classes_):
            full[:, list(classes).index(c)] = p[:, j]
        true += list(y.iloc[te]); pred += list(classes[full.argmax(1)]); probas.append(full)
    probas = np.vstack(probas)
    results[name] = dict(
        accuracy=accuracy_score(true, pred),
        f1_macro=f1_score(true, pred, average="macro"),
        log_loss=log_loss(true, np.clip(probas, 1e-6, 1), labels=classes),
    )
    print(f"\n=== {name} (leave-one-clip-out, {len(true)} held-out clips) ===")
    print({k: round(v, 3) for k, v in results[name].items()})
    print(classification_report(true, pred, zero_division=0))

pd.DataFrame(results).T.to_csv(os.path.join(HERE, "voice_model_results.csv"))
best = max(results, key=lambda n: (results[n]["f1_macro"], -results[n]["log_loss"]))
print("Best model:", best)

final = models[best].fit(X, y)           # refit on everything
THRESHOLD = 0.5                          # outsiders are rejected by the 'unknown' class

joblib.dump(dict(model=final, feature_cols=cols, threshold=THRESHOLD, classes=list(final.classes_)),
            os.path.join(HERE, "voice_model.joblib"))
print("saved voice_model.joblib  (classes:", list(final.classes_), ")")