# Letter sheets

One folder per letter, named `U<codepoint>_<letter>` (the codepoint keeps the
names safe on filesystems that mishandle Urdu characters).

| inside each folder | what it is | in git |
|---|---|---|
| `sheet.png` | every occurrence of that letter picked from the dataset videos, across signers and words. Written by `letter_reference.ipynb`. For the calibrated letters the calibration takes are appended at the bottom by `add_calibration_to_sheets.py`, one frame per take. ء is not in the dataset videos, so its sheet is the calibration panel alone. | yes |
| `user-hassan/` | individual calibration crops, one PNG per recorded frame. Hassan's own sessions: ا, ک, ب, ل. | yes |
| `user-mixed-abdullah-hassan/` | the same, for the 13 Nov session that Abdullah and Hassan recorded together (26 letters, 10 takes each, 20 crops per take). Who made which take was not recorded, so the crops are not under either name. | yes |
| `official/` | reference images of the official sign (screenshots from the PSL website), for checking the dataset labels against. | **no**, ignored |

`_alphabet.png` at the top level shows one exemplar per letter.

The labels behind `sheet.png` are inferred from the videos, not hand-checked, and
some sheets show more than one handshape. `official/` exists so each sheet can be
compared with the real sign.
