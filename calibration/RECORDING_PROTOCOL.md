# Recording new people (step 4 protocol, 23 Nov 2026)

Recordings of new people answer three open questions at once:

1. **Do the letter sheets still help once the clean data covers more people?** Each new person
   becomes a test round of `clean_eval.py` automatically (README #26, #29).
2. **Does calibration hold on a different day?** All 92–97% calibrated numbers so far come from
   the same sitting as the training takes.
3. **How many takes per letter does a new user need?** Compare 1, 3 and 5 takes.

## Before recording

- **Consent.** The repo is public. `add_calibration.py` puts the person's hand landmarks into
  `hold_landmarks.npy` (in git), and `add_calibration_to_sheets.py` would put crops of their
  hand, often with part of the face, into `letter_sheets/` (two frames per take in git). Tell
  each person this and ask. If they agree to landmarks only, do not run
  `add_calibration_to_sheets.py` for their file, and keep their `.json` (which holds crops) out
  of git.
- **Learn the signs from the official images**, not from each other or from the letter sheets
  (whose labels are often wrong). The team's `PSL.pdf` notes have the official image of every
  static letter except ء (screenshots of the PSL website; not in git). Easily confused pairs: ح and خ (two
  fingers sideways; خ has the thumb up), ک (index and middle up in a V) and گ.

## What to record

- **Only the 30 static letters:**
  ء ا ب پ ت ث چ ح خ د ڈ ذ ر ز ژ س ص ض ط ظ ع غ ف ق ک گ ل و ی ے.
  `web/capture.html` lists all 41; skip آ ؤ ئ ج ش م ن ٹ ڑ ھ ہ (signed with movement, out of scope).
- **5 takes per letter** in session 1. Lower the hand between takes so each take is formed afresh.
- **Shuffle the letter order** (not alphabetical), so neighbouring letters do not always follow
  each other.
- **Session 2, on a different day in a different room** (other light, other background): at
  least 3 takes per letter. Session 2 is the different-day test.
- **Ourselves too:** Hassan and Abdullah each record one session on a different day in a
  different room. This is the first different-day test of calibration (question 2).

## Naming

- File: keep the capture page's name, `calibration/calibration_<YYYY-MM-DD>.json`; rename if two
  sessions fall on one day (`..._ali.json`).
- Signer id: `user-<firstname>-<YYYY-MM-DD>`, lowercase, no spaces, e.g. `user-ali-2026-12-01`.
  The first name is what groups a person's sessions into one test round
  (`clean_eval.person_of`), so use the same first name for both sessions and never `hassan` or
  `abdullah` for anyone else. Our own new sessions: `user-hassan-<date>`, `user-abdullah-<date>`.

## After recording

    python add_calibration.py calibration/calibration_<date>.json --signer user-<name>-<date> --dry-run
    python add_calibration.py calibration/calibration_<date>.json --signer user-<name>-<date>
    PSL_KAMRAN_DIR=~/Desktop/train python clean_eval.py      # the person is now a round

`add_calibration.py` scores each take before merging and blocks takes whose letter looks like a
different one. For a new person, look at a blocked take before deciding: it can be a wrong
label, or the person's own way of signing that letter.

A person becomes a test round once they have at least 15 of the 30 letters
(`clean_eval.MIN_LETTERS`). The different-day test (question 2) and the takes-per-letter test
(question 3) need a short script each; write them when the first two-session recordings exist.
