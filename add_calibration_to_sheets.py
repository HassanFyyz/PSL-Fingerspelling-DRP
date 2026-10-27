"""
Extract the embedded hand crops from both calibration sessions as PNG files,
and append them to the existing alif/kahf letter reference sheets.

Usage: python add_calibration_to_sheets.py

Two outputs:
  1. letter_sheets/user-hassan/<codepoint>_<letter>/*.png -- every individual
     crop, one file per recorded frame (20 per take, 5 takes per letter per
     session = 100 per letter per session).
  2. letter_sheets/<codepoint>_<letter>.png -- the existing dataset reference
     sheet (built by letter_reference.ipynb), with a new panel appended below
     showing one representative frame per take of the developer's own
     recordings, clearly captioned and visually separated from the dataset
     rows above it rather than mixed in.

These crops are 64x64 grayscale and background-masked, the same format
capture.html and crop_dataset.py produce -- smaller and lower-resolution than
the dataset's own ~240x350 colour crops, so they are shown in their own panel
rather than blended into the existing grid.
"""
import base64
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SESSIONS = [
    ("calibration/calibration_2026-10-22.json", "session1"),
    ("calibration/calibration_2026-10-26.json", "session2"),
]
OUT_ROOT = "letter_sheets/user-hassan"
SHEET_DIR = "letter_sheets"
CROP_SIZE = 64


def decode_crop(b64str):
    raw = base64.b64decode(b64str)
    return np.frombuffer(raw, dtype=np.uint8).reshape(CROP_SIZE, CROP_SIZE)


def extract_all():
    """Save every embedded crop as its own PNG, grouped by letter."""
    by_letter = {}  # letter -> [(img, caption), ...]
    for path, tag in SESSIONS:
        d = json.load(open(path, encoding="utf-8"))
        for t in d["takes"]:
            letter = t["letter"]
            letter_dir = os.path.join(OUT_ROOT, f"U{ord(letter):04X}_{letter}")
            os.makedirs(letter_dir, exist_ok=True)
            for fi, fr in enumerate(t["frames"]):
                if fr.get("crop") is None:
                    continue
                img = decode_crop(fr["crop"])
                caption = f"{tag} take{t['take']} f{fi:02d}"
                name = f"{tag}_take{t['take']}_f{fi:02d}.png"
                Image.fromarray(img, mode="L").save(os.path.join(letter_dir, name))
                by_letter.setdefault(letter, []).append((img, caption, t["take"], tag))
    return by_letter


def append_to_sheet(letter, entries, ncol=5):
    """Append one representative frame per take to the existing dataset sheet."""
    sheet_path = os.path.join(SHEET_DIR, f"U{ord(letter):04X}_{letter}.png")
    if not os.path.exists(sheet_path):
        print(f"no existing sheet for {letter!r}, skipping append")
        return

    # one frame per (session, take) -- the middle of its 20 captured frames,
    # not all 20, which would just repeat the same settled pose
    by_take = {}
    for img, caption, take, tag in entries:
        by_take.setdefault((tag, take), []).append((img, caption))
    picks = []
    for (tag, take), frames in sorted(by_take.items()):
        img, caption = frames[len(frames) // 2]
        picks.append((img, caption))

    n = len(picks)
    ncol2 = min(ncol, n)
    nrow = int(np.ceil(n / ncol2))
    fig, axes = plt.subplots(nrow, ncol2, figsize=(2.05 * ncol2, 2.3 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (img, caption) in enumerate(picks):
        ax = axes[k // ncol2][k % ncol2]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(caption, fontsize=7.5)
    fig.suptitle(f"developer's own calibration recordings (64x64 masked, {n} takes shown)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    panel_path = sheet_path + ".usertmp.png"
    fig.savefig(panel_path, dpi=80, bbox_inches="tight")
    plt.close(fig)

    top = Image.open(sheet_path).convert("RGB")
    bottom = Image.open(panel_path).convert("RGB")
    w = max(top.width, bottom.width)
    top_r = top if top.width == w else top.resize((w, round(top.height * w / top.width)))
    bottom_r = bottom if bottom.width == w else bottom.resize((w, round(bottom.height * w / bottom.width)))
    combo = Image.new("RGB", (w, top_r.height + bottom_r.height), "white")
    combo.paste(top_r, (0, 0))
    combo.paste(bottom_r, (0, top_r.height))
    combo.save(sheet_path)
    os.remove(panel_path)
    print(f"appended {n} of your own takes to {sheet_path}")


if __name__ == "__main__":
    by_letter = extract_all()
    for letter, entries in by_letter.items():
        print(f"{letter}: extracted {len(entries)} crops to "
              f"{OUT_ROOT}/U{ord(letter):04X}_{letter}/")
        append_to_sheet(letter, entries)
