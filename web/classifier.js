/**
 * Letter classifier + lexicon decoder for the live demo.
 *
 * The model is HOG + multinomial logistic regression exported by
 * export_model.py. Inference is one matrix-vector multiply, so no runtime
 * needs shipping.
 *
 * READ THIS BEFORE TRUSTING ANY OUTPUT
 * ------------------------------------
 * Honest accuracy is 12.9 % leave-one-signer-out against a 9.8 % majority
 * baseline (lift 1.31x). On 40 classes that means **individual letters are
 * usually wrong** for a person the model never trained on, which is everyone
 * using this page. The per-letter output is shown for diagnosis, not because
 * it is expected to be right.
 *
 * The lexicon decoder exists because of that. Letters are unreliable
 * individually, but the whole predicted sequence can still be closer to the
 * correct word than to any other word in a 54-word vocabulary. Scoring every
 * candidate against the full per-letter probability distributions, rather than
 * against the argmax letters, uses the information in the runner-up guesses
 * instead of throwing it away.
 */

import { hog } from "./hog.js";
import * as wd from "./word_decode.js";

export class LetterClassifier {
  constructor(manifest, W, b) {
    this.manifest = manifest;
    this.classes = manifest.classes;
    this.nClasses = manifest.n_classes;
    this.nFeat = manifest.hog.n_features;
    this.imgSize = manifest.img_size;
    this.W = W;                 // Float32Array, nClasses * nFeat, row-major
    this.b = b;                 // Float32Array, nClasses
    this.lexicon = manifest.lexicon || {};
    this.classIndex = new Map(this.classes.map((c, i) => [c, i]));
  }

  static async load(base = "./model") {
    const manifest = await (await fetch(`${base}/letter_model.json`)).json();
    const W = new Float32Array(await (await fetch(`${base}/letter_w.bin`)).arrayBuffer());
    const b = new Float32Array(await (await fetch(`${base}/letter_b.bin`)).arrayBuffer());
    const m = new LetterClassifier(manifest, W, b);
    if (W.length !== m.nClasses * m.nFeat) {
      throw new Error(`weight size ${W.length} != ${m.nClasses}x${m.nFeat}`);
    }
    return m;
  }

  /** Softmax probabilities over letters for one grayscale crop. */
  predict(gray, size = this.imgSize) {
    const f = hog(gray, size);
    if (f.length !== this.nFeat) {
      throw new Error(`HOG length ${f.length} != expected ${this.nFeat}`);
    }
    const logits = new Float64Array(this.nClasses);
    for (let c = 0; c < this.nClasses; c++) {
      let s = this.b[c];
      const off = c * this.nFeat;
      for (let k = 0; k < this.nFeat; k++) s += this.W[off + k] * f[k];
      logits[c] = s;
    }
    let mx = -Infinity;
    for (const v of logits) if (v > mx) mx = v;
    let tot = 0;
    const p = new Float64Array(this.nClasses);
    for (let c = 0; c < this.nClasses; c++) { p[c] = Math.exp(logits[c] - mx); tot += p[c]; }
    for (let c = 0; c < this.nClasses; c++) p[c] /= tot;
    return p;
  }

  top(probs, k = 3) { return wd.top(this.classes, probs, k); }

  /** argmax letter sequence — what the model literally predicted. */
  rawWord(probSeq) { return wd.rawWord(this.classes, probSeq); }

  /**
   * Best lexicon match, scored against the full probability distributions.
   *
   * Only words whose length equals the number of detected holds are
   * considered. That is a real constraint rather than a trick: segmentation is
   * the reliable half of this pipeline (MAE ~1.2 letters), so hold count is
   * much better evidence than any individual letter. `slack` allows for the
   * segmenter being off by one.
   */
  decodeWord(probSeq, opts = {}) {
    return wd.decodeWord(this.classIndex, this.lexicon, probSeq, opts);
  }
}
