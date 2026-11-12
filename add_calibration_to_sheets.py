"""
Extract the embedded hand crops from the calibration sessions as PNG files,
and append them to the letter reference sheets.

Usage: python add_calibration_to_sheets.py <letter> [<letter> ...]
       e.g.  python add_calibration_to_sheets.py ب ل

Letters must be named explicitly. Each run APPENDS a panel to sheet.png, so
running it again for a letter that already has one adds a second copy. The
letters already done are listed in DONE below; the script refuses them unless
--again is passed.

Only takes that went into the training data are used (see SESSIONS): a
session's takes of a letter that was left out of training are not shown.

Two outputs:
  1. letter_sheets/<codepoint>_<letter>/user-hassan/*.png -- every individual
     crop, one file per recorded frame (20 per take, 5 takes per letter per
     session = 100 per letter per session).
  2. letter_sheets/<codepoint>_<letter>/sheet.png -- the existing dataset reference
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

# (file, tag used in crop names and captions, letters from that file that are
# in the training data -- None means all of them)
SESSIONS = [
    ("calibration/calibration_2026-10-22.json", "session1", None),
    ("calibration/calibration_2026-10-26.json", "session2", None),
    ("calibration/calibration_2026-11-11.json", "session3", {"\u0628"}),   # be only
    ("calibration/calibration_2026-11-12.json", "session4", {"\u0644"}),   # lam only
]
# letters whose sheet.png already carries the calibration panel
DONE = {"\u0627", "\u06a9", "\u0628", "\u0644"}
SHEET_DIR = "letter_sheets"
USER_DIR = "user-hassan"      # inside each letter's folder
CROP_SIZE = 64
# individual saved crops are upscaled for viewing -- 64x64 raw is too small to
# see clearly. LANCZOS rather than NEAREST since these are real recordings,
# not something where blockiness should be preserved as information.
SAVE_SIZE = 320


def decode_crop(b64str):
    raw = base64.b64decode(b64str)
    return np.frombuffer(raw, dtype=np.uint8).reshape(CROP_SIZE, CROP_SIZE)


def extract_all(letters):
    """Save every embedded crop of `letters` as its own PNG, grouped by letter."""
    by_letter = {}  # letter -> [(img, caption), ...]
    for path, tag, allowed in SESSIONS:
        d = json.load(open(path, encoding="utf-8"))
        for t in d["takes"]:
            letter = t["letter"]
            if letter not in letters or (allowed is not None and letter not in allowed):
                continue
            letter_dir = os.path.join(SHEET_DIR, f"U{ord(letter):04X}_{letter}", USER_DIR)
            os.makedirs(letter_dir, exist_ok=True)
            for fi, fr in enumerate(t["frames"]):
                if fr.get("crop") is None:
                    continue
                img = decode_crop(fr["crop"])
                caption = f"{tag} take{t['take']} f{fi:02d}"
                name = f"{tag}_take{t['take']}_f{fi:02d}.png"
                big = Image.fromarray(img, mode="L").resize((SAVE_SIZE, SAVE_SIZE), Image.LANCZOS)
                big.save(os.path.join(letter_dir, name))
                by_letter.setdefault(letter, []).append((img, caption, t["take"], tag))
    return by_letter


def append_to_sheet(letter, entries, ncol=5):
    """Append one representative frame per take to the existing dataset sheet."""
    sheet_path = os.path.join(SHEET_DIR, f"U{ord(letter):04X}_{letter}", "sheet.png")
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
    fig, axes = plt.subplots(nrow, ncol2, figsize=(4.1 * ncol2, 4.6 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (img, caption) in enumerate(picks):
        ax = axes[k // ncol2][k % ncol2]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        ax.set_title(caption, fontsize=13)
    fig.suptitle(f"developer's own calibration recordings (64x64 masked, {n} takes shown)",
                 fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    panel_path = sheet_path + ".usertmp.png"
    fig.savefig(panel_path, dpi=110, bbox_inches="tight")
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
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    already = [l for l in args if l in DONE]
    if already and "--again" not in sys.argv:
        sys.exit(f"{' '.join(already)} already has a calibration panel on its sheet "
                 "(see DONE). Pass --again to append another one anyway.")
    by_letter = extract_all(set(args))
    for letter, entries in by_letter.items():
        print(f"{letter}: extracted {len(entries)} crops to "
              f"{SHEET_DIR}/U{ord(letter):04X}_{letter}/{USER_DIR}/")
        append_to_sheet(letter, entries)
