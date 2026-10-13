# Verified claims from the three group papers

Every figure we cite from the published work, checked against the PDFs in this
repo. Nothing here is from memory or from a summary — each row was extracted
from the paper text directly.

**Use this file whenever writing up.** Numbers quoted from a paper go in with a
reference to a row below; numbers we computed ourselves go in with a reference
to the notebook that produced them.

Papers:
- **[SEG]** *Decoding the Speaking Hands of Silent Voices…* (segmentation paper)
- **[CFRT]** *PSL-CFRT: A Benchmark Multi-View Dataset…* (dataset paper)
- **[REAL]** *Toward Real-World Sign Language Recognition…* (SSRN 6308399)

---

## Headline accuracy figures

| Claim | Source | Verified wording |
|---|---|---|
| YOLO, front angle **67.84 %** | [SEG] Table | "on front angle videos the accuracy of YOLO model was 67.84%" |
| YOLO, 35–45° **48.07 %** | [SEG] | "tested on videos recorded at an angle of 35-45 degrees, the accuracy dropped significantly to 48.07%" |
| ResNet-50 **58.34 / 34.36** | [SEG] Table | front / tilted |
| VGG **40.72 / 22.93** | [SEG] Table | front / tilted |
| YOLOv8x baseline **63.16 %** (CER 36.84 %) | [CFRT] Table III | "YOLOv8x 36.84 % 63.16 %" |
| YOLOv5s **54.74 %** (CER 45.26 %) | [CFRT] Table III | |

Implied retention under angle change, YOLO: **48.07 / 67.84 = 0.709**.

## Dataset facts

| Claim | Source | Verified wording |
|---|---|---|
| **38** Urdu alphabet classes | [CFRT] | "categorized into 38 classes/objects/characters"; nouns "cover each of the 38 Urdu alphabets at least once" |
| **70 nouns** | [CFRT] | "the frequency of occurrence of each character in these **70 nouns** is listed in Figure 3" |
| every letter appears **≥5 times** | [CFRT] | "ensuring that each alphabet appears in the overall corpus at least five times" |
| 'ē' is the commonest letter, **455** occurrences | [CFRT] | "appeared 455 times in the dataset, making it the most frequently occurring Urdu character" |
| rarest letters appear **5 times** | [CFRT] | four letters "appeared only five times throughout the dataset" |
| annotated frames **2,913** | [CFRT] Table II | 2,040 train + 873 valid |
| annotated frames **2,913** | [SEG] Table I | 2,014 train + 899 valid — *same total, different split* |
| per-character images: min 27 / max 88 (train) | [CFRT], [SEG] | identical in both |

> **Discrepancy worth noting:** the two papers give the same grand total (2,913)
> but different train/val splits (2,040+873 vs 2,014+899). Not important for us,
> but worth being aware of if we ever cite the split.

> **Relevant to our canonicalization:** [CFRT] says **70 nouns**. Our corpus has
> **87 distinct filenames**, which canonicalize to **72** unique words
> (`decompose_letters.py`). Landing within 2 of the paper's figure is good
> independent corroboration that the canonicalization is close to correct — and
> suggests there may be **two residual merges** we have not caught. Worth asking.

## The SOFI algorithms, as actually specified

**Algorithm 1** [SEG] — fixed threshold:
```
while Optical flow intensity <= 2.5 do
    keep appending frames to make a segment
if length of segment > 9 then
    append this segment to the list of segments
```
Threshold **2.5**, minimum segment length **> 9 frames**. Our implementation
matches.

**Algorithm 2** [SEG] — dynamic:
```
 5: if intensity value decreases in subsequent frames then
 6:    keep tracking the lowest intensity value
 7:    if intensity value increases for fewer than 5 frames then
 8:        skip those values
11:    if intensity value decreases for more than 5 frames
       and intensity increases for more than 5 frames afterward then
12:        create a new video segment of 11 frames,
           centering around the lowest intensity frame
```
Prose: *"detecting stretches of at least five or more consecutive frames
exhibiting the least motion. … sudden spikes in motion spanning fewer than five
consecutive frames were ignored."*

> **IMPORTANT — correction to an earlier claim of ours.** We previously
> described the published spec as "≥5 strictly decreasing frames then ≥5
> increasing" and reported that this reading under-segments badly (5.7 fewer
> segments than letters). That is **unfair to the paper**: lines 7–9 explicitly
> skip brief increases, i.e. the algorithm already contains the noise tolerance
> our "strict" baseline omitted. Our strict variant was a strawman of our own
> construction, not the paper's method.
>
> What is fair to say: the pseudocode's "decreases for more than 5 frames"
> admits more than one implementation, and a strictly monotonic reading
> performs badly, so we implemented the windowed-minimum reading instead.
> **Do not present this as a flaw in the paper.**

Segment length **11 frames centred on the lowest-intensity frame** — our
`seg_len = 11` matches, and this is where that default comes from.

## [REAL] — the cross-dataset evaluation paper

| Claim | Verified wording |
|---|---|
| Three PSL corpora harmonized into **39 classes** | "harmonize three public Pakistani Sign Language (PSL) fingerspelling corpora into a unified 39-class (letter-level) dataset" |
| Protocol is **leave-one-dataset-out (LODO)** | as quoted above |
| Three metrics: **Generalization Gap, Retention Under Change, Robustness Tax** | "we report three operational metrics: Generalization Gap, Retention Under Change, and … Robustness Tax" |
| In-distribution accuracy ~**88.5–89.3 %** | 88.47 %, 89.28 % reported |
| LODO accuracy collapses to ~**20–45 %** | 20.2 %, 31.4 %, 39.3 %, 44.6 % reported |

We borrow **Retention Under Change** from this paper for `angle_evaluation.ipynb`.

---

## Our own numbers (for contrast — these come from notebooks, not papers)

| Figure | Notebook |
|---|---|
| Alg 1 MAE 4.28 / exact 5.96 % ; Alg 2 MAE 1.19 / exact 32.11 % | `sofi_algorithm2.ipynb` |
| LOSO held-out MAE 1.23 | `sofi_algorithm2.ipynb` |
| 218 clips, 14,530 hold frames, 40 classes | `pseudo_labeling.ipynb` |
| landmark letters 6.8 % vs 9.1 % majority; signer ID 97.0 % | `letter_classifier.ipynb` |
| frame-level 74.6 % vs occurrence-level 6.8 % = 10.9× inflation | `letter_classifier.ipynb` |
| raw pixels 8.3 % (0.85×), HOG 13.0 % (1.33×), HOG+mask 13.2 % (1.35×) | `letter_cnn.ipynb` |
| word top-1 33.1 %, top-3 55.8 %, chance 1.85 % | `letter_cnn.ipynb` |
| angle retention: recognition 0.690, segmentation 1.155 | `angle_evaluation.ipynb` |

Note our **40** letter classes vs the paper's **38**: the difference is three
hamza/madda forms (آ ؤ ئ) that we did not merge — see `decompose_letters.py`.
That is an open question for the authors.
