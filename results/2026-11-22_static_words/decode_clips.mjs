// Decode dumped clips with the page's own decoder (web/word_decode.js) and settings.
// Usage: node results/2026-11-22_static_words/decode_clips.mjs vote_clips.json
import { readFileSync } from "node:fs";
import { decodeAligned, MIN_HOLDS } from "../../web/word_decode.js";

const D = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = [];
for (const model of ["all", "static"]) {
  for (const c of D[model]) {
    const classIndex = new Map(c.classes.map((l, i) => [l, i]));
    const r = c.probs.length
      ? decodeAligned(classIndex, D.lexicon, c.probs, { wildLetters: D.dynamic_letters }) : [];
    const best = r[0];
    const shown = !!(best && best.shown);
    out.push({ model, signer: c.signer, word: c.word, n_holds: c.probs.length,
               best: best ? best.word : null,
               top1: best && best.word === c.word ? 1 : 0,
               top3: r.slice(0, 3).some(x => x.word === c.word) ? 1 : 0,
               agree: best ? best.agree : 0, n_scored: best ? best.nScored : 0,
               shown, shown_agree2: shown && best.agree >= 2, shown_agree3: shown && best.agree >= 3 });
  }
}
console.log(JSON.stringify(out));
