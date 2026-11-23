"""Cut the dataset tiles out of letter_sheets/*/sheet.png and read their captions (OCR).

Run from the repo root: python results/2026-11-23_flag_review/cut_tiles.py OUT_DIR
Needs tesseract (eng). Writes OUT_DIR/<codepoint>_<k>.png and OUT_DIR/tiles.csv.
Matching tiles to holds (signer + position, then the Urdu word) was done afterwards;
review_key.csv holds the result."""
import glob, os, re, sys, subprocess
import cv2, numpy as np, pandas as pd

REPO = "."
OUT = sys.argv[1] if len(sys.argv) > 1 else "tiles"; os.makedirs(OUT, exist_ok=True)

def tiles_of(path):
    img = cv2.imread(path)
    nonwhite = (img.min(axis=2) < 235).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nonwhite, 8)
    boxes = [tuple(s[:4]) for s in stats[1:] if s[4] > 30000 and s[2] > 150 and s[3] > 150]
    # colour tiles only (the calibration panel below is grey): mean saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    col = [b for b in boxes if hsv[b[1]:b[1]+b[3], b[0]:b[0]+b[2], 1].mean() > 40]
    col.sort(key=lambda b: (round(b[1] / 50), b[0]))       # reading order
    return img, col

def ocr(img):
    big = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    r = subprocess.run(["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6"],
                       input=cv2.imencode(".png", big)[1].tobytes(), capture_output=True)
    return r.stdout.decode("utf-8", "ignore")

rows = []
for sheet in sorted(glob.glob(f"{REPO}/letter_sheets/U*_*/sheet.png")):
    letter = os.path.basename(os.path.dirname(sheet)).split("_", 1)[1]
    img, boxes = tiles_of(sheet)
    for k, (x, y, w, h) in enumerate(boxes):
        cap = img[max(0, y - 75):y - 2, x - 20:x + w + 20]
        t = ocr(cap)
        sig = re.search(r"S\s*([1-5])\s*[-–]\s*([FLR])", t)
        pos = re.search(r"pos\s*(\d+)", t)
        f = f"{OUT}/{ord(letter):04X}_{k:02d}.png"
        cv2.imwrite(f, img[y:y+h, x:x+w])
        rows.append({"letter": letter, "tile": k, "file": f, "ocr": " ".join(t.split()),
                     "signer": f"signer{sig.group(1)}-{sig.group(2)}" if sig else None,
                     "pos": int(pos.group(1)) - 1 if pos else None, "low_trust": "low" in t})
T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "tiles.csv"), index=False)
print(len(T), "tiles;", T.signer.notna().mean().round(3), "signer read;", T.pos.notna().mean().round(3), "pos read")
print(T.groupby("letter").size().describe()[["min", "max"]].to_dict())
