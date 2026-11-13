"""
demo_check.py - tests verify_voice on clips the model has NEVER trained on.
Usage:  python demo_check.py blessing
(claims the identity 'blessing': his own clips should pass, demo_clips outsiders should fail)
"""
import os, sys, glob
from voice_verification import verify_voice

HERE = os.path.dirname(os.path.abspath(__file__))
claimed = sys.argv[1] if len(sys.argv) > 1 else "blessing"

def run(title, files):
    print(f"\n--- {title} (claiming to be '{claimed}') ---")
    ok = 0
    for f in files:
        r = verify_voice(f, claimed)
        ok += r["authorized"]
        print(f"{os.path.basename(f):38s} authorized={str(r['authorized']):5s} "
              f"conf={r['confidence']:.2f}  {r['reason']}")
    return ok, len(files)

genuine = sorted(glob.glob(os.path.join(HERE, "data", "audio", f"{claimed}_*.wav")))
outsiders = sorted(glob.glob(os.path.join(HERE, "data", "demo_clips", "*.wav")))

a, n = run("Genuine clips", genuine)
b, m = run("Unauthorized clips", outsiders)

print(f"\nGenuine accepted:      {a}/{n}")
print(f"Outsiders rejected:    {m - b}/{m}")
