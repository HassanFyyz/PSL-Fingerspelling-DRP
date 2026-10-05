/**
 * End-to-end check of the recognition path: real hold crops for a word ->
 * HOG -> logistic regression -> lexicon decode -> predicted word.
 *
 * This is the whole thing the live page does, minus the camera, so it can be
 * verified without a browser.
 *
 * Usage: node web/test_decode.mjs <word_fixture.json>
 */
import { readFileSync } from "node:fs";
import { LetterClassifier } from "./classifier.js";

function f32(file) {
  const buf = readFileSync(file);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

const manifest = JSON.parse(readFileSync("web/model/letter_model.json", "utf8"));
const clf = new LetterClassifier(manifest, f32("web/model/letter_w.bin"),
                                 f32("web/model/letter_b.bin"));

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
let top1 = 0, top3 = 0, letterHits = 0, letterTot = 0;

for (const c of cases) {
  const probs = c.grays.map(g => clf.predict(Float64Array.from(g), c.size));
  const raw = clf.rawWord(probs);
  const cands = clf.decodeWord(probs);
  const truth = manifest.lexicon[c.word] ? c.word : null;

  if (cands.length && cands[0].word === c.word) top1++;
  if (cands.slice(0, 3).some(x => x.word === c.word)) top3++;

  const trueLetters = manifest.lexicon[c.word] || [];
  for (let i = 0; i < Math.min(trueLetters.length, raw.length); i++) {
    letterTot++;
    if (raw[i] === trueLetters[i]) letterHits++;
  }
}

console.log(`words tested        : ${cases.length}`);
console.log(`per-letter accuracy : ${(100*letterHits/letterTot).toFixed(1)}%  (${letterHits}/${letterTot})`);
console.log(`word top-1          : ${(100*top1/cases.length).toFixed(1)}%  (${top1}/${cases.length})`);
console.log(`word top-3          : ${(100*top3/cases.length).toFixed(1)}%  (${top3}/${cases.length})`);
console.log(`lexicon size        : ${Object.keys(manifest.lexicon).length} words`);
