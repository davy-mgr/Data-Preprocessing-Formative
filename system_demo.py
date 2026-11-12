#!/usr/bin/env python3
"""
system_demo.py - Command-line demonstration of the full multimodal pipeline.

    Face image  -->  Voice approval  -->  Product recommendation

A transaction only reaches the product model if BOTH identity checks pass:

  Step 1  FACE   verify_face(image, member)    -> unlocks the transaction
  Step 2  VOICE  verify_voice(clip, member)    -> approves + executes prediction
  Step 3  MODEL  predict_product(customer_id)  -> recommendation is displayed

Usage
-----
  python system_demo.py                  # scripted demo: 1 authorized + 4 unauthorized
  python system_demo.py run  --member blessing --face PHOTO.jpg --voice CLIP.wav
  python system_demo.py interactive      # prompts for each input

Requires the packages in the Pipfile (torch, torchvision, librosa, scikit-learn...).
"""
import argparse
import contextlib
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACE_DIR = ROOT / "facial_recognition_model"
VOICE_DIR = ROOT / "voice_verification"
PRODUCT_DIR = ROOT / "product_model_project"

# ---------------------------------------------------------------------------
# EDIT ME: which customer record belongs to which team member.
# The datasets have no names, so this link is an assumption. These IDs were
# picked because they have several past transactions in merged_dataset.csv.
# ---------------------------------------------------------------------------
MEMBER_TO_CUSTOMER = {
    "blessing": 161,
    "davy": 159,
    "gabriella": 187,
    "kagaba": 188,
}

# ---------------------------------------------------------------------------
# Terminal styling (plain text when output is piped or NO_COLOR is set)
# ---------------------------------------------------------------------------
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def green(t): return _c(t, "32;1")
def red(t): return _c(t, "31;1")
def cyan(t): return _c(t, "36;1")
def dim(t): return _c(t, "2")


def banner(title):
    print("\n" + cyan("=" * 70))
    print(cyan(f"  {title}"))
    print(cyan("=" * 70))


def step(label, ok, detail):
    mark = green("PASS") if ok else red("FAIL")
    print(f"  [{mark}] {label:<22} {detail}")


# ---------------------------------------------------------------------------
# Loading the three existing modules (by file path, so the folder named
# "voice_verification" never shadows the module of the same name)
# ---------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _in_dir(path):
    """product_predictor.py reads its CSV/model with relative paths."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


_modules = {}


def face_module():
    if "face" not in _modules:
        try:
            _modules["face"] = _load_module("face_predictor", FACE_DIR / "face_predictor.py")
        except ModuleNotFoundError as err:
            sys.exit(f"Missing package '{err.name}'. Install the Pipfile dependencies first "
                     f"(pipenv install).")
    return _modules["face"]


def voice_module():
    if "voice" not in _modules:
        sys.path.insert(0, str(VOICE_DIR))
        _modules["voice"] = _load_module("voice_verification", VOICE_DIR / "voice_verification.py")
    return _modules["voice"]


def product_module():
    if "product" not in _modules:
        sys.path.insert(0, str(PRODUCT_DIR))
        _modules["product"] = _load_module("product_predictor", PRODUCT_DIR / "product_predictor.py")
    return _modules["product"]


# ---------------------------------------------------------------------------
# The transaction
# ---------------------------------------------------------------------------
def run_transaction(member, face_path, voice_path):
    """
    Run one transaction end to end. Returns a dict with the outcome:
        {"granted": bool, "stage": "face"|"voice"|"model", ...}
    """
    member = member.strip().lower()
    face_path, voice_path = Path(face_path), Path(voice_path)
    print(f"  Claimed identity : {member}")
    print(f"  Face image       : {face_path.name}")
    print(f"  Voice sample     : {voice_path.name}")
    print()

    for p in (face_path, voice_path):
        if not p.exists():
            step("Input files", False, f"file not found: {p}")
            return _deny("input", "input file not found")

    # ---- Step 1: face -----------------------------------------------------
    print(dim("  Step 1/3  Face verification"))
    from PIL import Image
    try:
        face = face_module().verify_face(Image.open(face_path), member)
    except KeyError:
        step("Face verification", False, f"'{member}' is not an enrolled member")
        return _deny("face", "not an enrolled member")
    except RuntimeError as err:
        step("Face verification", False, str(err))
        return _deny("face", str(err))

    step("Face verification", face["accepted"],
         f"similarity {face['similarity']:.3f} vs cutoff {face['cutoff']:.3f}")
    if not face["accepted"]:
        return _deny("face", "face does not match the claimed identity")

    print(f"  {green('Face accepted.')} Transaction unlocked - product model may now be called.")
    print(f"  {dim('Say your approval phrase to execute the prediction...')}\n")
    time.sleep(0.4)

    # ---- Step 2: voice ----------------------------------------------------
    print(dim("  Step 2/3  Voice approval"))
    voice = voice_module().verify_voice(str(voice_path), expected_user=member)
    step("Voice verification", voice["authorized"],
         f"confidence {voice['confidence']:.2f} - {voice['reason']}")
    if not voice["authorized"]:
        return _deny("voice", voice["reason"])

    print(f"  {green('Voice approved.')} Executing prediction...\n")
    time.sleep(0.4)

    # ---- Step 3: product model --------------------------------------------
    print(dim("  Step 3/3  Product recommendation"))
    customer_id = MEMBER_TO_CUSTOMER.get(member)
    if customer_id is None:
        step("Product model", False, f"no customer record linked to '{member}'")
        return _deny("model", "no customer record linked")
    try:
        with _in_dir(PRODUCT_DIR):
            result = product_module().predict_product(customer_id)
    except KeyError as err:
        step("Product model", False, str(err))
        return _deny("model", str(err))

    step("Product model", True, f"customer {customer_id}")
    print()
    print(green("  ACCESS GRANTED") + f"  -  recommended product for {member.title()}: "
          + green(result["top_prediction"]) + f"  ({result['confidence']:.0%} confidence)")
    for category, prob in result["top_k"]:
        bar = "#" * int(prob * 40)
        print(f"      {category:<12} {prob:6.1%}  {bar}")
    return {"granted": True, "stage": "model", "prediction": result}


def _deny(stage, reason):
    print()
    print(red("  ACCESS DENIED") + f"  -  stopped at the {stage} stage: {reason}")
    if stage in ("face", "voice", "input"):
        print(red("  The product model was NOT called."))
    return {"granted": False, "stage": stage, "reason": reason}


# ---------------------------------------------------------------------------
# Scripted demonstration
# ---------------------------------------------------------------------------
def _scenarios():
    img = FACE_DIR / "images"
    aud = VOICE_DIR / "data" / "audio"
    unauth = VOICE_DIR / "data" / "audio_unauthorized"
    clips = VOICE_DIR / "data" / "demo_clips"
    return [
        dict(title="1. AUTHORIZED transaction (expect: granted)",
             member="blessing", face=img / "blessing_neutral.jpg",
             voice=aud / "blessing_confirm_01.wav", expect=True),
        dict(title="2. UNAUTHORIZED IMAGE - Davy's face claiming to be Blessing (expect: denied at face)",
             member="blessing", face=img / "davy_smile.jpg",
             voice=aud / "blessing_confirm_01.wav", expect=False),
        dict(title="3. UNAUTHORIZED AUDIO - valid face, outsider's voice (expect: denied at voice)",
             member="blessing", face=img / "blessing_smile.jpg",
             voice=unauth / "kellia_yes_approve_01.wav", expect=False),
        dict(title="4. UNAUTHORIZED AUDIO - valid face, unseen outsider recording (expect: denied at voice)",
             member="blessing", face=img / "blessing_neutral.jpg",
             voice=clips / "keke_yes_approve_01.wav", expect=False),
        dict(title="5. VOICE MISMATCH - Gabriella's face, Kagaba's voice (expect: denied at voice)",
             member="gabriella", face=img / "gabriella_neutral.jpg",
             voice=aud / "kagaba_yesapprove_01.wav", expect=False),
    ]


def run_scripted_demo(pause):
    results = []
    for sc in _scenarios():
        banner(sc["title"])
        outcome = run_transaction(sc["member"], sc["face"], sc["voice"])
        results.append((sc["title"], sc["expect"], outcome["granted"]))
        if pause:
            input(dim("\n  Press Enter for the next scenario..."))

    banner("SUMMARY")
    all_ok = True
    for title, expected, granted in results:
        ok = expected == granted
        all_ok &= ok
        label = "granted" if granted else "denied "
        print(f"  [{green('OK') if ok else red('!!')}] {label}  {title.split(' (')[0]}")
    print()
    print(green("  All scenarios behaved as expected.") if all_ok
          else red("  Some scenarios did not behave as expected."))
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------
def run_interactive():
    banner("INTERACTIVE TRANSACTION")
    print("  Enrolled members:", ", ".join(sorted(MEMBER_TO_CUSTOMER)))
    print(dim("  (Ctrl+C to quit)\n"))
    try:
        while True:
            member = input("  Who are you claiming to be?   ").strip()
            face = input("  Path to face image:           ").strip().strip('"')
            voice = input("  Path to voice clip (.wav):    ").strip().strip('"')
            print()
            run_transaction(member, face, voice)
            if input("\n  Another transaction? [y/N] ").strip().lower() != "y":
                break
            print()
    except (KeyboardInterrupt, EOFError):
        print("\n  Bye.")
    return 0


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Face + voice gated product recommendation demo")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="scripted demo (default)")
    demo.add_argument("--pause", action="store_true", help="wait for Enter between scenarios")

    run = sub.add_parser("run", help="run one transaction with your own files")
    run.add_argument("--member", required=True)
    run.add_argument("--face", required=True, help="path to face image")
    run.add_argument("--voice", required=True, help="path to voice .wav")

    sub.add_parser("interactive", help="prompt for each input")
    sub.add_parser("members", help="list enrolled members and linked customer IDs")

    args = parser.parse_args()
    if args.command == "run":
        outcome = run_transaction(args.member, args.face, args.voice)
        return 0 if outcome["granted"] else 2
    if args.command == "interactive":
        return run_interactive()
    if args.command == "members":
        for member, customer_id in sorted(MEMBER_TO_CUSTOMER.items()):
            print(f"  {member:<12} customer {customer_id}")
        return 0
    return run_scripted_demo(pause=getattr(args, "pause", False))


if __name__ == "__main__":
    sys.exit(main())