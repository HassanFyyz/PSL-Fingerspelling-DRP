/**
 * Parity test: the browser HOG + logistic-regression path must reproduce the
 * Python one. A silent numerical drift here would make every prediction wrong
 * with no error message, so it is checked rather than assumed.
 *
 * Usage: node web/test_classifier.mjs <clf_fixture.json>
 */
import { readFileSync } from "node:fs";
import { hog } from "./hog.js";

const path = process.argv[2];
if (!path) { console.error("usage: node web/test_classifier.mjs <fixture.json>"); process.exit(2); }

const manifest = JSON.parse(readFileSync("web/model/letter_model.json", "utf8"));
// Node Buffers are views into a shared pool: .buffer alone ignores byteOffset
// and silently yields the wrong weights.
function f32(file) {
  const buf = readFileSync(file);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
const W = f32("web/model/letter_w.bin");
const b = f32("web/model/letter_b.bin");
const nC = manifest.n_classes, nF = manifest.hog.n_features;

const cases = JSON.parse(readFileSync(path, "utf8"));
let maxHog = 0, maxProb = 0, predMismatch = 0;

for (const c of cases) {
  const f = hog(Float64Array.from(c.gray), c.size);
  for (let i = 0; i < nF; i++) maxHog = Math.max(maxHog, Math.abs(f[i] - c.hog[i]));

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
  if (manifest.classes[am] !== c.pred) predMismatch++;
}

console.log(`cases                : ${cases.length}`);
console.log(`max |HOG diff|       : ${maxHog.toExponential(3)}`);
console.log(`max |prob diff|      : ${maxProb.toExponential(3)}`);
console.log(`argmax mismatches    : ${predMismatch}`);
const ok = maxHog < 1e-6 && maxProb < 1e-5 && predMismatch === 0;
console.log(ok ? "PARITY OK" : "PARITY FAILED");
process.exit(ok ? 0 : 1);
