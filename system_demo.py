import argparse
import contextlib
import gc
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACE_DIR = ROOT / "facial_recognition_model"
VOICE_DIR = ROOT / "voice_verification"
PRODUCT_DIR = ROOT / "product_model_project"

MEMBER_TO_CUSTOMER = {
    "blessing": 161,
    "davy": 159,
    "gabriella": 187,
    "kagaba": 188,
}

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


def _preview_face(face_path):
    from PIL import Image

    print(f"  Face preview   : {face_path}")
    with Image.open(face_path) as image:
        image.show()


def _preview_voice(voice_path):
    if sys.platform == "win32":
        import winsound

        print(f"  Voice playback: {voice_path}")
        winsound.PlaySound(str(voice_path), winsound.SND_FILENAME)
    else:
        print(dim("  Voice playback preview is only supported on Windows."))
    print(dim("  Compare the preview with the claimed identity."))


def _prompt_member():
    member = input("Registered username: ").strip().lower()
    if member not in MEMBER_TO_CUSTOMER:
        print(red("Unknown registered username."))
        return None
    return member


def _prompt_file(prompt):
    return Path(input(prompt).strip().strip('"'))


def _show_prediction(member, result):
    print(green(f"  Recommended product for {member.title()}: ")
          + green(result["top_prediction"])
          + f"  ({result['confidence']:.0%} confidence)")
    for category, probability in result["top_k"]:
        print(f"      {category:<12} {probability:6.1%}")


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


def _run_transaction(member, face_path, voice_path=None):
    """
    Run one transaction end to end. Returns a dict with the outcome:
        {"granted": bool, "stage": "face"|"voice"|"model", ...}
    """
    member = member.strip().lower()
    face_path = Path(face_path)
    print(f"  Claimed identity : {member}")
    print(f"  Face image       : {face_path.name}")
    print()

    if not face_path.is_file():
        step("Face image", False, f"file not found: {face_path}")
        return _deny("input", "face image not found")

    _preview_face(face_path)

    #Face verification
    print(dim("  Step 2/4  Facial recognition"))
    try:
        from PIL import Image
        with Image.open(face_path) as image:
            face = face_module().verify_face(image, member)
    except KeyError:
        step("Face verification", False, f"'{member}' is not an enrolled member")
        return _deny("face", "not an enrolled member")
    except (OSError, RuntimeError) as err:
        step("Face verification", False, str(err))
        return _deny("face", str(err))

    step("Face verification", face["accepted"],
         f"similarity {face['similarity']:.3f} vs cutoff {face['cutoff']:.3f}")
    if not face["accepted"]:
        return _deny("face", "face does not match the claimed identity")

    print(f"  {green('Face accepted.')} Continue to voice verification.")

    #Voice verification
    print(dim("  Step 3/4  Voice verification"))
    if voice_path is None:
        voice_path = input("  Submit voice recording path (.wav): ").strip().strip('"')
    voice_path = Path(voice_path)
    print(f"  Voice sample     : {voice_path.name}")
    if not voice_path.is_file():
        step("Voice sample", False, f"file not found: {voice_path}")
        return _deny("input", "voice sample not found")
    _preview_voice(voice_path)
    print(f"  {dim('Checking the voice recording...')}\n")
    time.sleep(0.4)

    try:
        voice = voice_module().verify_voice(str(voice_path), expected_user=member)
    except MemoryError as err:
        gc.collect()
        reason = f"insufficient memory during voice verification: {err}"
        step("Voice verification", False, reason)
        return {"granted": False, "stage": "error", "reason": reason}
    step("Voice verification", voice["authorized"],
         f"confidence {voice['confidence']:.2f} - {voice['reason']}")
    if not voice["authorized"]:
        return _deny("voice", voice["reason"])

    print(f"  {green('Voice approved.')} Executing prediction...\n")
    time.sleep(0.4)

    #Product prediction
    print(dim("  Step 4/4  Product prediction"))
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
    _show_prediction(member, result)
    return {"granted": True, "stage": "model", "prediction": result}


def run_transaction(member, face_path, voice_path=None):
    outcome = _run_transaction(member, face_path, voice_path)
    print("\n" + cyan("TRANSACTION COMPLETED"))
    if outcome["granted"]:
        result = outcome["prediction"]
        print(f"Final result: {result['top_prediction']} "
              f"({result['confidence']:.0%} confidence)")
    elif outcome["stage"] == "error":
        print(f"Final result: ERROR - {outcome['reason']}")
    else:
        print(f"Final result: denied at {outcome['stage']} stage - {outcome['reason']}")
    return outcome


def _deny(stage, reason):
    print()
    print(red("  ACCESS DENIED") + f"  -  stopped at the {stage} stage: {reason}")
    if stage in ("face", "voice", "input"):
        print(red("  The product model was NOT called."))
    return {"granted": False, "stage": stage, "reason": reason}


def _test_face_model():
    member = _prompt_member()
    if member is None:
        return
    image_path = _prompt_file("Face image path: ")
    if not image_path.is_file():
        print(red(f"Image not found: {image_path}"))
        return

    _preview_face(image_path)
    try:
        from PIL import Image
        with Image.open(image_path) as image:
            result = face_module().verify_face(image, member)
    except KeyError:
        print(red(f"'{member}' is not an enrolled member."))
        return
    except (OSError, RuntimeError) as err:
        print(red(str(err)))
        return

    step("Face verification", result["accepted"],
         f"similarity {result['similarity']:.3f} vs cutoff {result['cutoff']:.3f}")


def _test_voice_model():
    member = _prompt_member()
    if member is None:
        return
    voice_path = _prompt_file("Voice recording path (.wav): ")
    if not voice_path.is_file():
        print(red(f"Audio not found: {voice_path}"))
        return

    _preview_voice(voice_path)
    try:
        result = voice_module().verify_voice(str(voice_path), expected_user=member)
    except MemoryError as err:
        gc.collect()
        print(red(f"Voice verification could not complete: {err}"))
        return
    step("Voice verification", result["authorized"],
         f"speaker {result['predicted_user'] or 'unknown'}, "
         f"confidence {result['confidence']:.2f} - {result['reason']}")


def _test_product_model():
    member = _prompt_member()
    if member is None:
        return

    customer_id = MEMBER_TO_CUSTOMER[member]
    try:
        with _in_dir(PRODUCT_DIR):
            result = product_module().predict_product(customer_id)
    except KeyError as err:
        print(red(str(err)))
        return
    _show_prediction(member, result)

#Scripted demonstration

def _scenarios():
    img = FACE_DIR / "images"
    aud = VOICE_DIR / "data" / "audio"
    unauth = VOICE_DIR / "data" / "audio_unauthorized"
    clips = VOICE_DIR / "data" / "demo_clips"
    return [
        dict(title="1. AUTHORIZED transaction (expect: granted)",
             member="blessing", face=img / "blessing_neutral.jpg",
             voice=aud / "blessing_confirm_01.wav", expect=True),
        dict(title="2. AUTHORIZED transaction - Davy (expect: granted)",
             member="davy", face=img / "davy_neutral.jpg",
             voice=aud / "davy_confirm_01.wav", expect=True),
        dict(title="3. UNAUTHORIZED IMAGE - Davy's face claiming to be Blessing (expect: denied at face)",
             member="blessing", face=img / "davy_smile.jpg",
             voice=aud / "blessing_confirm_01.wav", expect=False),
        dict(title="4. UNAUTHORIZED AUDIO - valid face, outsider's voice (expect: denied at voice)",
             member="blessing", face=img / "blessing_smile.jpg",
             voice=unauth / "kellia_yes_approve_01.wav", expect=False),
        dict(title="5. UNAUTHORIZED AUDIO - valid face, unseen outsider recording (expect: denied at voice)",
             member="blessing", face=img / "blessing_neutral.jpg",
             voice=clips / "keke_yes_approve_01.wav", expect=False),
        dict(title="6. VOICE MISMATCH - Gabriella's face, Kagaba's voice (expect: denied at voice)",
             member="gabriella", face=img / "gabriella_neutral.jpg",
             voice=aud / "kagaba_yesapprove_01.wav", expect=False),
    ]


def run_scripted_demo(pause):
    results = []
    for sc in _scenarios():
        banner(sc["title"])
        outcome = run_transaction(sc["member"], sc["face"], sc["voice"])
        results.append((sc["title"], sc["expect"], outcome))
        if pause:
            input(dim("\n  Press Enter for the next scenario..."))

    banner("FINAL RESULTS")
    all_ok = True
    for title, expected, outcome in results:
        error = outcome.get("stage") == "error"
        granted = outcome["granted"]
        ok = not error and expected == granted
        all_ok &= ok
        expected_status = "AUTHORIZED" if expected else "UNAUTHORIZED"
        actual_status = "ERROR" if error else (
            "AUTHORIZED" if granted else "UNAUTHORIZED"
        )
        status = green("PASS") if ok else red("FAIL")
        print(f"  {title.split(' (')[0]}: expected {expected_status}, "
              f"result {actual_status} [{status}]")

    authorized = sum(expected for _, expected, _ in results)
    unauthorized = len(results) - authorized
    authorized_passed = sum(expected and outcome["granted"] and outcome.get("stage") != "error"
                            for _, expected, outcome in results)
    unauthorized_blocked = sum(not expected and not outcome["granted"]
                               and outcome.get("stage") != "error"
                               for _, expected, outcome in results)
    errors = sum(outcome.get("stage") == "error" for _, _, outcome in results)
    print()
    print(f"  Authorized: {authorized_passed}/{authorized} passed")
    print(f"  Unauthorized: {unauthorized_blocked}/{unauthorized} blocked")
    if errors:
        print(red(f"  System errors: {errors}"))
    print(green("  All scenarios behaved as expected.") if all_ok
          else red("  Some scenarios did not behave as expected."))
    return 0 if all_ok else 1

#Interactive mode

def run_interactive():
    banner("MODEL TEST APP")
    print("  Registered members:", ", ".join(sorted(MEMBER_TO_CUSTOMER)))
    try:
        while True:
            print("\n  1. Face recognition")
            print("  2. Voice verification")
            print("  3. Product prediction")
            print("  4. Full transaction")
            print("  5. Run all demo scenarios")
            print("  0. Exit")
            choice = input("Choose a model or action: ").strip()
            if choice == "0":
                break
            if choice == "1":
                _test_face_model()
            elif choice == "2":
                _test_voice_model()
            elif choice == "3":
                _test_product_model()
            elif choice == "4":
                member = _prompt_member()
                if member is not None:
                    run_transaction(member, _prompt_file("Face image path: "))
            elif choice == "5":
                run_scripted_demo(pause=False)
            else:
                print(red("Choose an option from 0 to 5."))
    except (KeyboardInterrupt, EOFError):
        print("\n  Bye.")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Face + voice gated product recommendation demo")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run all scripted scenarios")
    demo.add_argument("--pause", action="store_true", help="wait for Enter between scenarios")

    run = sub.add_parser("run", help="run one transaction with your own files")
    run.add_argument("--member", required=True)
    run.add_argument("--face", required=True, help="path to face image")
    run.add_argument("--voice", required=True, help="path to voice .wav")

    sub.add_parser("interactive", help="choose and test models interactively")
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
    if args.command == "demo":
        return run_scripted_demo(pause=args.pause)
    return run_interactive()


if __name__ == "__main__":
    sys.exit(main())