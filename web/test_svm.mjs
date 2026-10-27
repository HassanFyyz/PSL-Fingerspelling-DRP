/**
 * Parity test: the browser hand-geometry + SVM-OVO decode must reproduce
 * sklearn's SVC.predict() exactly. This is the check that export_geometry_model.py
 * already ran in pure Python before anything was ported -- re-run here against
 * the actual JS so a future edit to either side cannot drift silently.
 *
 * Usage: node web/test_svm.mjs web/svm_fixture.json
 */
import { readFileSync } from "node:fs";
import { handFeatures } from "./hand_geometry.js";

const path = process.argv[2] || "web/svm_fixture.json";
const manifest = JSON.parse(readFileSync("web/model_geom/geom_model.json", "utf8"));

function f32(file) {
  const buf = readFileSync(file);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
const sv = f32("web/model_geom/geom_sv.bin");
const dualCoef = f32("web/model_geom/geom_dualcoef.bin");
const intercept = f32("web/model_geom/geom_intercept.bin");
const mean = f32("web/model_geom/geom_scaler_mean.bin");
const scale = f32("web/model_geom/geom_scaler_scale.bin");

const nFeat = manifest.n_feat, nSv = manifest.n_sv, nClasses = manifest.n_classes, gamma = manifest.gamma;
const start = new Int32Array(nClasses + 1);
for (let c = 0; c < nClasses; c++) start[c + 1] = start[c] + manifest.n_support[c];

function predictOne(pts) {
  const raw = handFeatures(pts);
  const x = new Float64Array(nFeat);
  for (let i = 0; i < nFeat; i++) x[i] = (raw[i] - mean[i]) / scale[i];

  const K = new Float64Array(nSv);
  for (let s = 0; s < nSv; s++) {
    let d2 = 0; const off = s * nFeat;
    for (let k = 0; k < nFeat; k++) { const d = sv[off + k] - x[k]; d2 += d * d; }
    K[s] = Math.exp(-gamma * d2);
  }

  const votes = new Float64Array(nClasses);
  let p = 0;
  for (let i = 0; i < nClasses; i++) {
    for (let j = i + 1; j < nClasses; j++) {
      const si = start[i], ei = start[i + 1], sj = start[j], ej = start[j + 1];
      let val = 0;
      const rowJ = (j - 1) * nSv, rowI = i * nSv;
      for (let s = si; s < ei; s++) val += dualCoef[rowJ + s] * K[s];
      for (let s = sj; s < ej; s++) val += dualCoef[rowI + s] * K[s];
      val += intercept[p];
      if (val > 0) votes[i] += 1; else votes[j] += 1;
      p++;
    }
  }
  let best = 0;
  for (let c = 1; c < nClasses; c++) if (votes[c] > votes[best]) best = c;
  return manifest.classes[best];
}

const cases = JSON.parse(readFileSync(path, "utf8"));
let mismatches = 0;
for (const c of cases) {
  const pred = predictOne(c.pts);
  if (pred !== c.pred) {
    mismatches++;
    console.log(`mismatch: true=${c.letter} python=${c.pred} js=${pred}`);
  }
}

console.log(`cases                : ${cases.length}`);
console.log(`argmax mismatches    : ${mismatches}`);
const ok = mismatches === 0;
console.log(ok ? "PARITY OK" : "PARITY FAILED");
process.exit(ok ? 0 : 1);
