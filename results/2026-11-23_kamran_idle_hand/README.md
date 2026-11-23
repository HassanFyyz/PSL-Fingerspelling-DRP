# Kamran's images: the detector often found the idle hand (23 Nov 2026)

**Found while checking why ک did not transfer between our calibration takes and Kamran's set.**

Kamran's images are full-body shots, 1920×1080: one hand signs at chest or face height, the
other hangs at the signer's side. The landmark cache was made with a one-hand detector
(`num_hands=1`), and in **1,348 of the 2,902 usable images (46%) it picked the idle hand**.
Those rows describe a resting hand with the fingers pointing down, whatever the letter.

**Evidence.** Height of the detected hand's centre (pixels from the top, 1080 = bottom):

| y | images |
|---|---|
| 100–700 (chest and face) | 1,554 |
| **700–900** | **0** |
| 900–1083 (at the hip) | 1,348, every one with fingertips below the wrist |

Of 12 random images above 900 viewed by eye, all 12 showed someone signing with one hand
while the detected hand hung at their side; all 6 random images below 700 had the signing
hand detected. The gap makes the cut exact: `image_dataset.IDLE_Y = 800`.

**What it explains.**
- The "ث sink" in step 2: idle hands all look alike, and models trained on our takes mapped
  them to ث (59 ا, 48 و, 41 ت … images read as ث).
- ک "not transferring": 51 of Kamran's 80 usable ک images were idle hands. Without them his
  ک is read as ک 18 of 29 times by the sheets-trained model.
- Round A's judge "flagging 54% of correct labels": 17.8% once the idle hands are gone.

**What changed.**
- `image_dataset.load_usable()` drops these rows (and missing hands, duplicate files);
  `clean_eval.py`, `flag_sheets.py`, `cleaned_eval.py` and `export_geometry_model.py
  --with-kamran` all use it. Usable: 1,554 images, 1,211 of the 30 static letters (22–74 per letter).
- Steps 2 and 3 were re-run (their READMEs give the corrected numbers).
- **Every earlier result that used Kamran's data included these rows**: README #24 (our models
  on his images 35.2 / 29.3%, the private model's 35.7%, "adding his images helps") and the first
  versions of #26–#28. The private browser model `web/model_geom_kamran/` contains them and
  should be rebuilt: `python export_geometry_model.py --with-kamran --with-official`.

**Getting the 1,348 images back.** `python image_dataset.py` re-detects with up to two hands and
keeps the higher one (it backs up the old cache first). It needs MediaPipe's
`hand_landmarker.task`, which this environment could not download; run it on a machine that can.
Not yet run, so not yet checked.

No images from Kamran's set are stored in this folder.
