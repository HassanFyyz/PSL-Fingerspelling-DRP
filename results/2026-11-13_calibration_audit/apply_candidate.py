"""Install the evaluated candidate locally, after checking the frozen baseline.

Run from the repository root. No training, Git operations, or backup overwrites.
The complete prior data and model remain in training_before/ and model_before/.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pairs = [(OUT / "training_after" / name, ROOT / name,
              OUT / "training_before" / name)
             for name in ("hold_landmarks.npy", "hold_meta.csv")]
    pairs += [(src, ROOT / "web/model_geom" / src.name,
               OUT / "model_before" / src.name)
              for src in sorted((OUT / "model_after").iterdir())]
    # Check every file before touching any. Allow an identical re-run.
    for src, dest, before in pairs:
        if sha(dest) not in (sha(before), sha(src)):
            raise RuntimeError(f"Unexpected local changes in {dest}; refusing to replace")
    manifest = {"copied": [], "backup": str(OUT),
                "source": "calibration/calibration_2026-11-13.json",
                "mixed_recorders": ["Abdullah", "Hassan"],
                "take_identity": "unknown; do not treat as a single new person",
                "row_policy": "one median landmark frame per take; all 261 takes kept",
                "class_policy": "41 classes, adding standalone hamza ء as requested"}
    for src, dest, before in pairs:
        shutil.copy2(src, dest)
        manifest["copied"].append({"path": str(dest.relative_to(ROOT)),
                                   "before_sha256": sha(before), "after_sha256": sha(dest)})
    (OUT / "installation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Installed candidate training data and model. Prior versions preserved in", OUT)


if __name__ == "__main__":
    main()
