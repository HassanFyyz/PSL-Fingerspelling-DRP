# Letter sheets

One folder per letter, named `U<codepoint>_<letter>` (the codepoint keeps the
names safe on filesystems that mishandle Urdu characters).

| inside each folder | what it is | in git |
|---|---|---|
| `sheet.png` | every occurrence of that letter picked from the dataset videos, across signers and words. Written by `letter_reference.ipynb`. For the calibrated letters the developer's own takes are appended at the bottom by `add_calibration_to_sheets.py`. | yes |
| `user-hassan/` | individual calibration crops, one PNG per recorded frame. Only for letters that have been calibrated. | yes |
| `official/` | reference images of the official sign (screenshots from the PSL website), for checking the dataset labels against. | **no**, ignored |

`_alphabet.png` at the top level shows one exemplar per letter.

The labels behind `sheet.png` are inferred from the videos, not hand-checked, and
some sheets show more than one handshape. `official/` exists so each sheet can be
compared with the real sign.
