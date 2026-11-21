# Letter sheets

One folder per letter, named `U<codepoint>_<letter>` (the codepoint keeps the
names safe on filesystems that mishandle Urdu characters).

| inside each folder | what it is | in git |
|---|---|---|
| `sheet.png` | every occurrence of that letter picked from the dataset videos, across signers and words. Written by `letter_reference.ipynb`. For the calibrated letters the calibration takes are appended at the bottom by `add_calibration_to_sheets.py`, one frame per take. ء is not in the dataset videos, so its sheet is the calibration panel alone. | yes |
| `user-hassan/` | individual calibration crops, one PNG per recorded frame. Hassan's own sessions: ا, ک, ب, ل. | yes |
| `user-mixed-abdullah-hassan/` | the same, for the 13 Nov session that Abdullah and Hassan recorded together (26 letters, 10 takes each, 20 crops per take). Who made which take was not recorded, so the crops are not under either name. | **only frames 5 and 15 of each take** (522 of 5,220, about 12.5 MB); the other 18 per take are local only, see below |
| `official/` | reference images of the official sign (screenshots from the PSL website), for checking the dataset labels against. | **no**, ignored |

`_alphabet.png` at the top level shows one exemplar per letter.

**The 13 Nov crops are only partly in git.** All 20 crops per take would add
about 125 MB to the repository for good, so `.gitignore` keeps only
`*_f05.png` and `*_f15.png`. The full set is on Hassan's machine and can be
rebuilt by anyone from the session file:

    python add_calibration_to_sheets.py

which reads `calibration/calibration_2026-11-13.json` (in git). Nothing is lost
by not committing them; do not "fix" the ignore rule without deciding about the
repository size first.

The labels behind `sheet.png` are inferred from the videos, not hand-checked, and
some sheets show more than one handshape. `official/` exists so each sheet can be
compared with the real sign.
