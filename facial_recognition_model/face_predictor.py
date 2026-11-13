import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
FEATURES_PATH = HERE / "image_features.csv"
CUTOFF = 0.534

_cache = {}


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _cosine(left, right):
    return float(np.dot(_unit(left), _unit(right)))


def _load():
    if "model" in _cache:
        return _cache["model"], _cache["detector"], _cache["preprocess"], _cache["stored"]

    model = torch.hub.load(
        "otroshi/edgeface",
        "edgeface_xxs",
        source="github",
        pretrained=True,
        trust_repo=True,
    )
    model.eval()

    repo = Path(torch.hub.get_dir()) / "otroshi_edgeface_main"
    sys.path.insert(0, str(repo))
    from face_alignment.mtcnn import MTCNN

    detector = MTCNN(device="cpu", crop_size=(112, 112))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    features = pd.read_csv(FEATURES_PATH)
    embedding_columns = [column for column in features.columns if column.startswith("emb_")]
    enrollment = features[
        features["expression"].isin(["neutral", "smile"]) & features["augmentation"].eq("original")
    ]
    stored = {
        member: _unit(group[embedding_columns].mean().to_numpy())
        for member, group in enrollment.groupby("member")
    }

    _cache.update(
        model=model,
        detector=detector,
        preprocess=preprocess,
        stored=stored,
    )
    return model, detector, preprocess, stored


def _embed(image, model, detector, preprocess):
    _boxes, faces = detector.align_multi(image.convert("RGB"), limit=1)
    if not faces:
        raise RuntimeError("No face detected in the photograph.")
    tensor = preprocess(faces[0]).unsqueeze(0)
    with torch.no_grad():
        vector = model(tensor).squeeze(0).cpu().numpy()
    return vector.astype(np.float64)


def verify_face(image, member):
    """Accept the claim when the photograph matches the member's stored embedding."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image.")

    model, detector, preprocess, stored = _load()
    name = str(member).strip().lower()
    if name not in stored:
        raise KeyError(f"Unknown member: {member}")

    similarity = _cosine(_embed(image, model, detector, preprocess), stored[name])
    return {
        "member": name,
        "accepted": similarity >= CUTOFF,
        "similarity": round(similarity, 3),
        "cutoff": CUTOFF,
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(__file__).name} <member> <photo>")
        sys.exit(1)
    claimed, photo = sys.argv[1], sys.argv[2]
    try:
        result = verify_face(Image.open(photo), claimed)
    except (KeyError, RuntimeError, TypeError, FileNotFoundError) as error:
        print(error.args[0] if error.args else error)
        sys.exit(1)
    decision = "accepted" if result["accepted"] else "rejected"
    print(
        f"Claim {result['member'].title()}: {decision} "
        f"(similarity {result['similarity']:.3f}, cutoff {result['cutoff']:.3f})"
    )
