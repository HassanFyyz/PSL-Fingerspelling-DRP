/**
 * Behaviour test for the alignment decoder in word_decode.js, on the real
 * lexicon. Builds vote-share-like score vectors for a known word and checks:
 *
 *   1. clean signing            -> that word, shown
 *   2. one spurious hold        -> still that word, shown
 *   3. one missed hold          -> still that word, shown
 *   4. one or two holds         -> nothing shown
 *   5. letters matching no word -> nothing shown
 *   6. the old decoder fails case 2 (the reason this decoder exists)
 *
 * Usage: node web/test_align.mjs
 */
import { readFileSync } from "node:fs";
import { decodeAligned, decodeWord, MIN_HOLDS } from "./word_decode.js";

const manifest = JSON.parse(readFileSync("web/model_geom/geom_model.json", "utf8"));
const classes = manifest.classes, lexicon = manifest.lexicon;
const classIndex = new Map(classes.map((c, i) => [c, i]));
const K = classes.length;

/** A hold whose top letter is `letter`: vote share ~0.95 for it, low elsewhere. */
function hold(letter, seed) {
  const p = new Float64Array(K);
  for (let c = 0; c < K; c++) p[c] = 0.30 + 0.25 * (((c * 7 + seed * 13) % 11) / 11);
  p[classIndex.get(letter)] = 0.95;
  return p;
}
const seq = letters => letters.map((l, i) => hold(l, i));

let failed = 0;
function check(name, cond, detail = "") {
  console.log(`${cond ? "ok  " : "FAIL"}  ${name}${detail ? "   " + detail : ""}`);
  if (!cond) failed++;
}

// a word long enough to lose or gain a hold
const [word, letters] = Object.entries(lexicon).find(([, ls]) => ls.length === 5);
const other = classes.find(c => !letters.includes(c));

let r = decodeAligned(classIndex, lexicon, seq(letters));
check("clean signing picks the word", r[0].word === word && r[0].shown, `${r[0].agree}/${r[0].nLetters} letters match`);

const extra = [...letters.slice(0, 2), other, ...letters.slice(2)];
r = decodeAligned(classIndex, lexicon, seq(extra));
check("one spurious hold: still the word", r[0].word === word && r[0].shown, `${r[0].agree}/${r[0].nLetters}`);

const old = decodeWord(classIndex, lexicon, seq(extra));
check("old decoder fails on the spurious hold", !old.length || old[0].word !== word,
      old.length ? `old decoder said a different word` : "old decoder had no candidate");

const missing = [...letters.slice(0, 2), ...letters.slice(3)];
r = decodeAligned(classIndex, lexicon, seq(missing));
check("one missed hold: still the word", r[0].word === word && r[0].shown, `${r[0].agree}/${r[0].nLetters}`);

for (let n = 1; n < MIN_HOLDS; n++) {
  r = decodeAligned(classIndex, lexicon, seq(letters.slice(0, n)));
  check(`${n} hold(s): nothing shown`, r.every(c => !c.shown));
}

// Letters that support no dictionary word: every hold has the same top 3, chosen
// so that no word has 60% of its letters among them.
function garbageHold(top3) {
  const p = new Float64Array(K);
  for (let c = 0; c < K; c++) p[c] = 0.10 + 0.001 * c;          // distinct, all low
  top3.forEach((l, i) => { p[classIndex.get(l)] = 0.95 - 0.05 * i; });
  return p;
}
let triple = null;
search:
for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) for (let c = b + 1; c < K; c++) {
  const t = new Set([classes[a], classes[b], classes[c]]);
  if (Object.values(lexicon).every(ls => ls.filter(l => t.has(l)).length / ls.length < 0.6)) {
    triple = [...t]; break search;
  }
}
check("found three letters that support no word", triple !== null);
r = decodeAligned(classIndex, lexicon, Array.from({ length: 5 }, () => garbageHold(triple)));
check("letters matching no word: nothing shown", r.every(c => !c.shown),
      `best support ${(100 * Math.max(...r.map(c => c.support))).toFixed(0)}%`);

console.log(failed ? `\n${failed} check(s) FAILED` : "\nALL OK");
process.exit(failed ? 1 : 0);
