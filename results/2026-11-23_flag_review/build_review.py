"""
Step 3, point 4: build the BLIND review page for the sheet-crop flags.

Inputs
  review_key.csv        the 80 items (40 the round-B judge dropped, 40 it kept, shuffled),
                        each with its label, the judge's best other letter, and which side
                        the label is shown on. Made from the sheet tiles that could be tied
                        to one hold for certain (cut_tiles.py: tiles cut from
                        letter_sheets/*/sheet.png, captions read by OCR, 8 of 8 spot-checked).
  tiles/                the cut tile images (cut_tiles.py writes them)
  official/             one official PSL screenshot per letter (from PSL.pdf; NOT in git)

Output
  review_page.html      self-contained; git-ignored because it embeds the official screenshots.
                        The reviewer sees each crop beside two letters (label and alternative,
                        random sides) and picks left / right / neither / can't tell. Nothing on
                        the page says which crops were flagged. "Export answers" downloads a CSV;
                        score it with score_review.py.

    python results/2026-11-23_flag_review/build_review.py TILES_DIR OFFICIAL_DIR
"""
import base64
import glob
import io
import json
import os
import sys

import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def b64(img, max_w, max_h, q=82):
    img = img.convert("RGB")
    img.thumbnail((max_w, max_h))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ours(letter):
    fs = sorted(glob.glob(f"{REPO}/letter_sheets/U{ord(letter):04X}_*/user-*/*_f05.png")) or \
        sorted(glob.glob(f"{REPO}/letter_sheets/U{ord(letter):04X}_*/user-hassan/*.png"))
    return b64(Image.open(fs[0]), 150, 150) if fs else None


def official(letter, off_dir):
    fs = glob.glob(os.path.join(off_dir, f"*_{letter}.png"))
    if not fs:
        return None
    im = Image.open(fs[0])
    w, h = im.size
    return b64(im.crop((int(w * 0.12), int(h * 0.05), int(w * 0.50), int(h * 0.80))), 300, 260)


def main(tiles_dir, off_dir):
    K = pd.read_csv(os.path.join(HERE, "review_key.csv"))
    letters = sorted(set(K.letter) | set(K.alt))
    refs = {l: {"official": official(l, off_dir), "ours": ours(l)} for l in letters}
    items = []
    for r in K.itertuples():
        left, right = (r.letter, r.alt) if r.label_side == "left" else (r.alt, r.letter)
        items.append({"item": f"{r.item:02d}" if isinstance(r.item, int) else str(r.item),
                      "crop": b64(Image.open(os.path.join(tiles_dir, r.tile_file)), 260, 330),
                      "left": left, "right": right})
    html = PAGE.replace("__DATA__", json.dumps({"items": items, "refs": refs}, ensure_ascii=False))
    out = os.path.join(HERE, "review_page.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"wrote {out}: {len(items)} items, {os.path.getsize(out) / 1e6:.1f} MB")


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Letter-sheet crop review (blind)</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --ink:#1d1d1f; --muted:#6b6b70; --line:#dcdce0; --bg:#f6f6f4; --pick:#1f6feb; }
  * { box-sizing: border-box; }
  body { margin:0; font:15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); }
  header { position:sticky; top:0; z-index:2; background:#fff; border-bottom:1px solid var(--line);
           padding:12px 20px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:17px; margin:0; }
  #progress { color:var(--muted); }
  button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:8px; padding:7px 12px; cursor:pointer; }
  button.primary { background:var(--ink); color:#fff; border-color:var(--ink); }
  main { max-width:1000px; margin:0 auto; padding:16px 20px 60px; }
  .intro { background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:16px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:14px;
          display:grid; grid-template-columns: 260px 1fr; gap:16px; }
  .card.done { border-color:#9fbf9f; }
  .crop img { width:100%; border-radius:6px; display:block; }
  .num { color:var(--muted); font-size:13px; margin-bottom:6px; }
  .choices { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  .choice { border:2px solid var(--line); border-radius:8px; padding:8px; cursor:pointer; text-align:center; }
  .choice.sel { border-color:var(--pick); background:#eef4ff; }
  .glyph { font-size:40px; line-height:1.1; font-family:"Noto Nastaliq Urdu","Noto Naskh Arabic","Geeza Pro",serif; }
  .refs { display:flex; gap:6px; justify-content:center; align-items:flex-start; margin-top:6px; }
  .refs img { height:150px; border-radius:4px; }
  .refs .cap { font-size:11px; color:var(--muted); }
  .other { grid-column:1 / -1; display:flex; gap:8px; }
  .other button.sel { border-color:var(--pick); background:#eef4ff; }
  @media (max-width:720px) { .card { grid-template-columns:1fr; } }
</style></head><body>
<header><h1>Which letter does each crop show?</h1><span id="progress"></span>
  <button class="primary" id="export">Export answers (CSV)</button></header>
<main>
  <div class="intro">Each crop is a frame cut from a dataset video and labelled with a letter.
  For each one, pick the letter whose sign the hand shape matches. Each choice shows the
  official PSL image and one of our own calibration takes. Pick <b>Neither</b> if it matches
  neither, and <b>Can't tell</b> if the frame is too blurred or in between two signs.
  Judge the hand shape only. Answers are kept in this browser while you work; press
  <b>Export answers</b> at the end and send the CSV back.</div>
  <div id="list"></div>
</main>
<script>
const DATA = __DATA__;
const KEY = "psl-flag-review-v1";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { answers = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {} render(); }
function refHtml(l) {
  const r = DATA.refs[l] || {};
  let h = "";
  if (r.official) h += `<div><img src="${r.official}" alt="official ${l}"><div class="cap">official</div></div>`;
  if (r.ours) h += `<div><img src="${r.ours}" alt="our take ${l}"><div class="cap">our take</div></div>`;
  return h || `<div class="cap">no reference image</div>`;
}
function render() {
  const list = document.getElementById("list");
  if (!list.childElementCount) {
    for (const it of DATA.items) {
      const c = document.createElement("div"); c.className = "card"; c.id = "c" + it.item;
      c.innerHTML = `<div class="crop"><div class="num">Crop ${it.item} of ${DATA.items.length}</div>
        <img src="${it.crop}" alt="crop ${it.item}"></div>
        <div><div class="choices">
          <div class="choice" data-v="left"><div class="glyph">${it.left}</div><div class="refs">${refHtml(it.left)}</div></div>
          <div class="choice" data-v="right"><div class="glyph">${it.right}</div><div class="refs">${refHtml(it.right)}</div></div>
          <div class="other"><button data-v="neither">Neither</button><button data-v="unsure">Can't tell</button></div>
        </div></div>`;
      c.querySelectorAll("[data-v]").forEach(el => el.addEventListener("click", () => { answers[it.item] = el.dataset.v; save(); }));
      list.appendChild(c);
    }
  }
  let n = 0;
  for (const it of DATA.items) {
    const a = answers[it.item]; if (a) n++;
    const c = document.getElementById("c" + it.item);
    c.classList.toggle("done", !!a);
    c.querySelectorAll("[data-v]").forEach(el => el.classList.toggle("sel", el.dataset.v === a));
  }
  document.getElementById("progress").textContent = `${n} of ${DATA.items.length} answered`;
}
document.getElementById("export").addEventListener("click", () => {
  const rows = ["item,left,right,answer"];
  for (const it of DATA.items) rows.push(`${it.item},${it.left},${it.right},${answers[it.item] || ""}`);
  const blob = new Blob(["﻿" + rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "flag_review_answers.csv";
  document.body.appendChild(a); a.click(); a.remove();
});
render();
</script></body></html>
"""

if __name__ == "__main__":
    main(*sys.argv[1:3])
