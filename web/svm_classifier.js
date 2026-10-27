/**
 * Hand-geometry + RBF-SVM letter classifier for the live demo.
 *
 * Replaces the HOG + logistic-regression model: 27.0% leave-one-signer-out
 * vs 12.9% for HOG+LogReg (hand_geometry.py). Features are 74-dim joint
 * angles + palm-normalised distances from MediaPipe landmarks, not image
 * crops -- see hand_geometry.js.
 *
 * This replays sklearn's libsvm one-vs-one multiclass decode by hand: for
 * every class pair, a signed kernel-sum margin decides a winner, and the
 * class with the most pairwise wins is the prediction (ties go to the lowest
 * class index -- confirmed against sklearn, NOT a margin-sum tiebreak, which
 * does not match). See export_geometry_model.py for how this was validated
 * before being ported, and test_svm.mjs for the parity check that re-verifies
 * it here.
 *
 * There is no Platt-scaled probability (that needs porting libsvm's pairwise
 * coupling too). Instead each class's "probability" is its fraction of
 * pairwise wins out of n_classes-1 -- not calibrated, but monotonic with
 * confidence and good enough for top-k display and lexicon scoring.
 */
import { handFeatures } from "./hand_geometry.js";
import * as wd from "./word_decode.js";

export class SvmClassifier {
  constructor(manifest, sv, dualCoef, intercept, scalerMean, scalerScale) {
    this.manifest = manifest;
    this.classes = manifest.classes;
    this.nClasses = manifest.n_classes;
    this.nFeat = manifest.n_feat;
    this.nSv = manifest.n_sv;
    this.gamma = manifest.gamma;
    this.sv = sv;                 // Float32Array, n_sv * n_feat, row-major
    this.dualCoef = dualCoef;     // Float32Array, (n_classes-1) * n_sv, row-major
    this.intercept = intercept;   // Float32Array, n_pairs
    this.scalerMean = scalerMean;     // Float32Array, n_feat
    this.scalerScale = scalerScale;   // Float32Array, n_feat
    this.lexicon = manifest.lexicon || {};
    this.classIndex = new Map(this.classes.map((c, i) => [c, i]));

    // class -> [start, end) row range into sv / columns of dualCoef
    this.start = new Int32Array(this.nClasses + 1);
    for (let c = 0; c < this.nClasses; c++) {
      this.start[c + 1] = this.start[c] + manifest.n_support[c];
    }
  }

  static async load(base = "./model_geom") {
    const manifest = await (await fetch(`${base}/geom_model.json`)).json();
    const f32 = async (name) =>
      new Float32Array(await (await fetch(`${base}/${name}`)).arrayBuffer());
    const [sv, dualCoef, intercept, mean, scale] = await Promise.all([
      f32("geom_sv.bin"), f32("geom_dualcoef.bin"), f32("geom_intercept.bin"),
      f32("geom_scaler_mean.bin"), f32("geom_scaler_scale.bin"),
    ]);
    const m = new SvmClassifier(manifest, sv, dualCoef, intercept, mean, scale);
    if (sv.length !== m.nSv * m.nFeat) throw new Error(`sv size ${sv.length} != ${m.nSv}x${m.nFeat}`);
    if (dualCoef.length !== (m.nClasses - 1) * m.nSv) {
      throw new Error(`dual_coef size ${dualCoef.length} != ${m.nClasses - 1}x${m.nSv}`);
    }
    return m;
  }

  /** Standardised 74-dim feature vector from one frame's 21 [x,y,z] landmarks. */
  _features(pts) {
    const raw = handFeatures(pts);
    const out = new Float64Array(this.nFeat);
    for (let i = 0; i < this.nFeat; i++) out[i] = (raw[i] - this.scalerMean[i]) / this.scalerScale[i];
    return out;
  }

  /**
   * Fraction of pairwise one-vs-one wins per class (0..1), for ONE frame's
   * landmarks. Sums to more than 1 across classes (it is not a true
   * probability) but argmax matches sklearn's SVC.predict() exactly.
   */
  _voteShare(pts) {
    const x = this._features(pts);
    const { sv, dualCoef, intercept, nFeat, nSv, nClasses, gamma, start } = this;

    const K = new Float64Array(nSv);
    for (let s = 0; s < nSv; s++) {
      let d2 = 0;
      const off = s * nFeat;
      for (let k = 0; k < nFeat; k++) { const d = sv[off + k] - x[k]; d2 += d * d; }
      K[s] = Math.exp(-gamma * d2);
    }

    const votes = new Float64Array(nClasses);
    let p = 0;
    for (let i = 0; i < nClasses; i++) {
      for (let j = i + 1; j < nClasses; j++) {
        const si = start[i], ei = start[i + 1];
        const sj = start[j], ej = start[j + 1];
        let val = 0;
        // class i's SVs use dual_coef row (j-1); class j's SVs use row i
        const rowJ = (j - 1) * nSv, rowI = i * nSv;
        for (let s = si; s < ei; s++) val += dualCoef[rowJ + s] * K[s];
        for (let s = sj; s < ej; s++) val += dualCoef[rowI + s] * K[s];
        val += intercept[p];
        if (val > 0) votes[i] += 1; else votes[j] += 1;
        p++;
      }
    }
    const share = new Float64Array(nClasses);
    for (let c = 0; c < nClasses; c++) share[c] = votes[c] / (nClasses - 1);
    return share;
  }

  /**
   * Prediction for a hold: average the vote-share distribution over every
   * frame given (same rationale as calibrate.py's take_vote -- average
   * per-frame distributions rather than trusting a single frame).
   *
   * `ptsFrames` is an array of frames, each 21 [x,y,z] isotropic landmarks.
   */
  predict(ptsFrames) {
    const frames = Array.isArray(ptsFrames[0][0]) ? ptsFrames : [ptsFrames];
    const acc = new Float64Array(this.nClasses);
    for (const pts of frames) {
      const s = this._voteShare(pts);
      for (let c = 0; c < this.nClasses; c++) acc[c] += s[c];
    }
    for (let c = 0; c < this.nClasses; c++) acc[c] /= frames.length;
    return acc;
  }

  top(probs, k = 3) { return wd.top(this.classes, probs, k); }
  rawWord(probSeq) { return wd.rawWord(this.classes, probSeq); }
  decodeWord(probSeq, opts = {}) { return wd.decodeWord(this.classIndex, this.lexicon, probSeq, opts); }
}
