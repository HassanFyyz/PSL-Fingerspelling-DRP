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
- `image_dataset.load_usable()` drops idle-hand rows (and missing hands, duplicate files);
  `clean_eval.py`, `flag_sheets.py`, `cleaned_eval.py` and `export_geometry_model.py
  --with-kamran` all use it. Usable: 1,554 images, 1,211 of the 30 static letters (22–74 per letter).
- Steps 2 and 3 were re-run (their READMEs give the corrected numbers).
- **Every earlier result that used Kamran's data included these rows**: README #24 (our models
  on his images 35.2 / 29.3%, the private model's 35.7%, "adding his images helps") and the first
  versions of #26–#28. The private browser model `web/model_geom_kamran/` contains them and
  should be rebuilt: `python export_geometry_model.py --with-kamran --with-official`.

**Getting the images back (done, 23 Nov).** `PSL_KAMRAN_DIR=~/Desktop/train python image_dataset.py`
re-detected all 2,952 images with up to two hands, keeping the higher one (MediaPipe 1.2.0, on
Abdullah's Mac). The one-hand cache is kept beside it as `landmarks.npy.one_hand_backup`.

| | count |
|---|---|
| hand found | 2,952 of 2,952 |
| previously idle hand, now the signing hand | 1,316 |
| still idle (signing hand not found; dropped by `load_usable`) | 48 |
| previously right, same hand kept (landmarks within 20 px) | 1,532 of 1,577 |
| previously right, moved more | 45 |
| **usable now** | **2,863 (2,322 of the static letters, 54–119 per letter)** |

Checked by eye on 12 images with the old and new skeletons drawn: in all 8 recovered images the
new landmarks sit on the signing hand and the old ones on the hanging hand. Of 4 "moved" images,
3 are the same signing hand fitted slightly differently and 1 was an idle hand the old cut had
missed (that signer's resting hand sits higher), now the signing hand.

**Copies of Kamran's folder elsewhere still hold the one-hand cache.** Copy `landmarks.npy` and
`landmarks_meta.csv` from Abdullah's folder, or run `python image_dataset.py` there.

No images from Kamran's set are stored in this folder.
