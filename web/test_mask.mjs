/**
 * Parity test for hand masking: the browser must mask the crop the same way
 * crop_dataset.hand_mask does, or the masked model receives input it was never
 * trained on and every prediction is wrong with no error shown.
 *
 * cv2's fillConvexPoly+dilate and the JS distance test are not bit-identical
 * at the mask boundary, so this checks that the PREDICTIONS agree, which is
 * what actually matters.
 *
 * Usage: node web/test_mask.mjs <mask_fixture.json>
 */
import { readFileSync } from "node:fs";
import { applyHandMask } from "./handmask.js";
import { hog } from "./hog.js";

function f32(file) {
  const buf = readFileSync(file);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
const manifest = JSON.parse(readFileSync("web/model/letter_model.json", "utf8"));
const W = f32("web/model/letter_w.bin"), b = f32("web/model/letter_b.bin");
const nC = manifest.n_classes, nF = manifest.hog.n_features;

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
let agree = 0, pixDiff = 0, maxProb = 0;

for (const c of cases) {
  const gray = Float64Array.from(c.gray);
  applyHandMask(gray, c.pts, c.box, c.size);

  let diff = 0;
  for (let i = 0; i < gray.length; i++) if (Math.abs(gray[i] - c.masked_gray[i]) > 1) diff++;
  pixDiff += diff / gray.length;

  const f = hog(gray, c.size);
  const logits = new Float64Array(nC);
  for (let k = 0; k < nC; k++) {
    let s = b[k];
    for (let i = 0; i < nF; i++) s += W[k * nF + i] * f[i];
    logits[k] = s;
  }
  let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v);
  let tot = 0; const p = new Float64Array(nC);
  for (let k = 0; k < nC; k++) { p[k] = Math.exp(logits[k] - mx); tot += p[k]; }
  for (let k = 0; k < nC; k++) p[k] /= tot;
  for (let k = 0; k < nC; k++) maxProb = Math.max(maxProb, Math.abs(p[k] - c.probs[k]));

  let am = 0; for (let k = 1; k < nC; k++) if (p[k] > p[am]) am = k;
  if (manifest.classes[am] === c.pred) agree++;
}

console.log(`cases                 : ${cases.length}`);
console.log(`mean masked-pixel diff: ${(100*pixDiff/cases.length).toFixed(2)}%  (boundary only)`);
console.log(`max |prob diff|       : ${maxProb.toExponential(2)}`);
console.log(`prediction agreement  : ${agree}/${cases.length}`);
const ok = agree === cases.length;
console.log(ok ? "MASK PARITY OK" : "MASK PARITY FAILED");
process.exit(ok ? 0 : 1);
