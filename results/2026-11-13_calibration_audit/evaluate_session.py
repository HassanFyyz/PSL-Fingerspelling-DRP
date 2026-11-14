"""Fixed-recipe calibration experiment; reads snapshots and never updates the app.

Usage: .venv/bin/python .../evaluate_session.py REPO CAPTURE OUTPUT
Before running, create OUTPUT/training_before/{hold_landmarks.npy,hold_meta.csv}
and OUTPUT/model_before/ (the frozen browser model). Fits the existing recipe;
does not tune settings, split frames across folds, or assign unknown identities.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fit(F, y):
    return make_pipeline(StandardScaler(), SVC(
        C=10, gamma="scale", class_weight="balanced",
        decision_function_shape="ovo", probability=False, random_state=0)).fit(F, y)


def shares(model, F):
    """Browser-compatible OVO wins; NOT Platt probabilities or OVR scores."""
    n = len(model.classes_)
    result = []
    for start in range(0, len(F), 512):
        margins = model.decision_function(F[start:start + 512])
        votes = np.zeros((len(margins), n))
        p = 0
        for i in range(n):
            for j in range(i + 1, n):
                win = margins[:, p] > 0
                votes[:, i] += win
                votes[:, j] += ~win
                p += 1
        result.append(votes / (n - 1))
    return np.concatenate(result)


def per_letter(rows, destination):
    r = pd.DataFrame(rows)
    r.to_csv(destination / "take_predictions.csv", index=False)
    table = r.groupby(["letter", "model"], sort=False).agg(
        n=("correct", "size"), correct=("correct", "sum"), recall=("correct", "mean"))
    table.to_csv(destination / "take_per_letter.csv")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("capture", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve()))
    from hand_geometry import hand_features

    started = time.monotonic()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads(a.capture.read_text())
    P = np.load(out / "training_before/hold_landmarks.npy")
    M = pd.read_csv(out / "training_before/hold_meta.csv")
    assert len(P) == len(M)
    F, y = hand_features(P), M.letter.to_numpy()
    takes = d["takes"]
    ids = [(t["letter"], t["take"]) for t in takes]
    assert len(ids) == len(set(ids)), "Duplicate take IDs"
    frames = [np.array([f["pts"] for f in t["frames"]], dtype=np.float32) for t in takes]
    assert all(p.shape[1:] == (21, 3) and np.isfinite(p).all() for p in frames)
    medianP = np.stack([p[len(p) // 2] for p in frames])
    Fu = hand_features(medianP)
    yu = np.array([t["letter"] for t in takes])
    allF = [hand_features(p) for p in frames]
    classes = sorted(set(y) | set(yu))
    assert len(classes) == 41 and "ء" in set(yu) - set(y)
    # Round-robin by within-letter take order: entire takes, never frames.
    counts = {}
    fold = []
    for t in takes:
        l = t["letter"]
        fold.append(counts.get(l, 0) % 5)
        counts[l] = counts.get(l, 0) + 1
    fold = np.array(fold)
    pd.DataFrame([{"letter": t["letter"], "take": t["take"], "fold": int(k),
                   "recorders": "Abdullah and Hassan; take identity unknown",
                   "median_frame": len(p) // 2, "n_frames": len(p),
                   "handedness_recorded": t.get("handedness")}
                  for t, k, p in zip(takes, fold, frames)]).to_csv(out / "take_folds.csv", index=False)
    base = json.loads((out / "shipped_exact_predictions.json").read_text())
    assert [(r["letter"], r["take"]) for r in base["rows"]] == ids
    rows = [{"letter": r["letter"], "take": r["take"], "fold": int(k),
             "model": "before_frozen_browser", "pred": r["pred"],
             "correct": r["correct"], "top3": " | ".join(x["letter"] for x in r["top3"])}
            for r, k in zip(base["rows"], fold)]
    for k in range(5):
        train, test = fold != k, np.where(fold == k)[0]
        model = fit(np.vstack([F, Fu[train]]), np.concatenate([y, yu[train]]))
        lengths = [len(allF[i]) for i in test]
        v = shares(model, np.concatenate([allF[i] for i in test]))
        start = 0
        for i, n in zip(test, lengths):
            avg = v[start:start + n].mean(0)
            start += n
            order = sorted(range(len(avg)), key=lambda j: (-avg[j], j))
            pred = str(model.classes_[order[0]])
            rows.append({"letter": yu[i], "take": takes[i]["take"], "fold": k,
                         "model": "after_same_session_5fold", "pred": pred,
                         "correct": bool(pred == yu[i]),
                         "top3": " | ".join(model.classes_[j] for j in order[:3])})
        print(f"Same-session fold {k + 1}/5: {len(test)} whole takes tested", flush=True)
    r = per_letter(rows, out)
    session_summary = {}
    for name, group in r.groupby("model"):
        session_summary[name] = {"n_takes": len(group), "n_correct": int(group.correct.sum()),
                                 "accuracy": float(group.correct.mean()),
                                 "macro_recall_26_recorded_letters": float(group.groupby("letter").correct.mean().mean())}
        cm = confusion_matrix(group.letter, group.pred, labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"take_confusion_{name}.csv")

    # Original 5 signers remain test subjects, including when training on the new session.
    dataset = ~M.signer.str.startswith("user-").to_numpy()
    names = sorted(set(M.signer[dataset]))
    predictions, metrics = [], []
    for signer in names:
        test = M.signer.to_numpy() == signer
        for mode in ("dataset_only", "previous_calibration", "plus_new_session"):
            train = ~test & (dataset if mode == "dataset_only" else np.ones(len(M), dtype=bool))
            Xtr, ytr = F[train], y[train]
            if mode == "plus_new_session":
                Xtr = np.vstack([Xtr, Fu])
                ytr = np.concatenate([ytr, yu])
            model = fit(Xtr, ytr)
            pred = model.predict(F[test])
            metrics.append({"signer": signer, "model": mode, "n": int(test.sum()),
                            "correct": int((pred == y[test]).sum()), "accuracy": float((pred == y[test]).mean()),
                            "macro_recall_present_test_classes": float(recall_score(y[test], pred, labels=sorted(set(y[test])), average="macro", zero_division=0))})
            for i, p in zip(np.where(test)[0], pred):
                predictions.append({"occ": M.iloc[i].occ, "signer": signer, "letter": y[i],
                                    "model": mode, "pred": str(p), "correct": bool(p == y[i])})
        print(f"Dataset LOSO {signer}: completed all three conditions", flush=True)
    pd.DataFrame(metrics).to_csv(out / "dataset_loso_per_signer.csv", index=False)
    D = pd.DataFrame(predictions)
    D.to_csv(out / "dataset_loso_predictions.csv", index=False)
    D.groupby(["letter", "model"]).agg(n=("correct", "size"), correct=("correct", "sum"), recall=("correct", "mean")).to_csv(out / "dataset_loso_per_letter.csv")
    dataset_summary = {}
    for mode, group in D.groupby("model"):
        cm = confusion_matrix(group.letter, group.pred, labels=classes)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"dataset_confusion_{mode}.csv")
        dataset_summary[mode] = {"n": len(group), "n_correct": int(group.correct.sum()),
                                "pooled_accuracy": float(group.correct.mean()),
                                "mean_signer_accuracy": float(group.groupby("signer").correct.mean().mean()),
                                "macro_recall_40_dataset_letters": float(group.groupby("letter").correct.mean().mean())}

    # Prepare a deployment candidate without replacing the root training artifacts.
    signer_id = "user-mixed-abdullah-hassan-2026-11-13"
    assert signer_id not in set(M.signer)
    newM = pd.DataFrame([{"signer": signer_id, "video": f"calib_{t['letter']}_{t['take']}",
                          "letter": t["letter"], "occ": f"{signer_id}|calib_{t['letter']}_{t['take']}|0"} for t in takes])
    dst = out / "training_after"
    dst.mkdir(exist_ok=True)
    np.save(dst / "hold_landmarks.npy", np.concatenate([P, medianP]))
    pd.concat([M, newM], ignore_index=True).to_csv(dst / "hold_meta.csv", index=False)
    final = fit(np.vstack([F, Fu]), np.concatenate([y, yu]))
    svc, scaler = final.named_steps["svc"], final.named_steps["standardscaler"]
    dest = out / "model_after"
    dest.mkdir(exist_ok=True)
    for file, array in {"geom_sv.bin": svc.support_vectors_, "geom_dualcoef.bin": svc.dual_coef_,
                        "geom_intercept.bin": svc.intercept_, "geom_scaler_mean.bin": scaler.mean_,
                        "geom_scaler_scale.bin": scaler.scale_}.items():
        array.astype(np.float32).tofile(dest / file)
    manifest = json.loads((out / "model_before/geom_model.json").read_text())
    manifest.update(n_classes=len(svc.classes_), classes=[str(c) for c in svc.classes_],
                    n_sv=len(svc.support_vectors_), n_support=svc.n_support_.astype(int).tolist(),
                    gamma=float(svc._gamma), n_pairs=len(svc.intercept_),
                    train_holds=len(P) + len(medianP), train_signers=sorted(set(M.signer) | {signer_id}))
    manifest["honest_accuracy"] = {
        "protocol": "Five original dataset test signers, one held out at a time. Training includes other four dataset signers plus 30 prior Hassan rows and all 261 new mixed-session rows. Test labels are pseudo-labels; test has 40 letters, classifier has 41. Mixed session is never a claimed signer-held-out fold.",
        "letter_acc": dataset_summary["plus_new_session"]["mean_signer_accuracy"],
        "note": "Same-session calibration and transfer-to-dataset results are separate; no measured new-session or live accuracy after this update."}
    manifest["new_calibration"] = {"file": str(a.capture), "sha256": hashlib.sha256(a.capture.read_bytes()).hexdigest(),
                                    "n_takes": len(takes), "training_rows_per_take": 1,
                                    "recorders": ["abdullah", "hassan"], "take_identity": "unknown",
                                    "new_class": "ء", "reported_physical_hand": "right",
                                    "mediapipe_take_hands": dict(__import__('collections').Counter(t.get('handedness') for t in takes))}
    (dest / "geom_model.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    # Include all median frames and a sample of hold-out frames in the candidate parity fixture.
    probeP = np.concatenate([P, medianP])
    pred = final.predict(hand_features(probeP))
    fixture = [{"pts": p.tolist(), "pred": str(z)} for p, z in zip(probeP, pred)]
    (out / "candidate_fixture.json").write_text(json.dumps(fixture, ensure_ascii=False))
    version = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
               "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    summary = {"versions": version, "source_sha256": hashlib.sha256(a.capture.read_bytes()).hexdigest(),
               "settings": {"C": 10, "gamma": "scale", "class_weight": "balanced", "probability": False},
               "session": session_summary, "dataset": dataset_summary, "elapsed_s": time.monotonic() - started,
               "limitations": ["mixed known people, unknown per-take identity", "one sitting", "same-session take CV is optimistic", "dataset test labels are pseudo-labels", "40-to-41-class change", "no new-person or new-day test", "training on one median frame differs from averaging live frames"]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
