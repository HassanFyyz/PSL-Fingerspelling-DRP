"""
Train / validation / test splits, defined once so every notebook uses the same rows.

There are two different questions, and each needs its own split.

1. NEW PERSON (cross-signer). Rows from the five dataset signers. A signer is
   never divided: all of one signer's holds sit in exactly one of train,
   validation or test. A random split of holds would put the same person on
   both sides and report how well the model recognises people it has seen.
   With five signers a single fixed split would test on one person, so the
   split is ROTATED: each signer is the test signer once, the next signer in
   name order is the validation signer, and the other three are training.

       rotation   train                 validation   test
       1          signers 3, 4, 5       signer 2     signer 1
       2          signers 1, 4, 5       signer 3     signer 2
       ...

   Settings and models are chosen on the validation signer only. The test
   signer is scored once, after the choice is made.

2. CALIBRATED USER. Rows from the calibration recordings (`user-*`). Whole
   takes are assigned, stratified by letter: 60% train, 20% validation, 20%
   test (10 takes -> 6 / 2 / 2, 5 takes -> 3 / 1 / 1). The five dataset
   signers are all training data here. The people in the test takes also made
   training takes in the same sitting, so this measures recognition of a
   calibrated user in the session they calibrated in, nothing wider.

`split_table()` writes both into one table (`data_splits.csv`).
"""
import numpy as np
import pandas as pd

USER_PREFIX = "user-"

# Letters whose sign needs movement. The pipeline classifies one held hand
# shape, so these are outside what it can recognise (a stated limitation).
# They are the 11 letters with no calibration takes.
DYNAMIC_LETTERS = tuple("آ ؤ ئ ج ش م ن ٹ ڑ ھ ہ".split())


def static_only(P, M):
    """(P, M) without the rows of the dynamic letters."""
    keep = ~M.letter.isin(DYNAMIC_LETTERS).to_numpy()
    return P[keep], M[keep].reset_index(drop=True)


def is_user(signers):
    return pd.Series(signers).astype(str).str.startswith(USER_PREFIX).to_numpy()


def signer_rotations(signers):
    """[(train_signers, val_signer, test_signer), ...], one per dataset signer."""
    names = sorted(set(s for s in signers if not str(s).startswith(USER_PREFIX)))
    out = []
    for i, test in enumerate(names):
        val = names[(i + 1) % len(names)]
        train = [s for s in names if s not in (val, test)]
        out.append((train, val, test))
    return out


def calibration_split(M, seed=0, frac=(0.6, 0.2, 0.2)):
    """Role of every row: 'train' / 'val' / 'test' for calibration rows, '' otherwise.

    One row of `M` is one take. Takes are shuffled within each letter with a
    fixed seed, so the assignment is reproducible and stratified by letter.
    """
    rng = np.random.default_rng(seed)
    user = is_user(M.signer)
    role = np.array([""] * len(M), dtype=object)
    for letter in sorted(set(M.letter[user])):
        idx = np.where(user & (M.letter.to_numpy() == letter))[0]
        idx = rng.permutation(idx)
        n = len(idx)
        n_test = max(1, int(round(frac[2] * n)))
        n_val = max(1, int(round(frac[1] * n)))
        role[idx[:n_test]] = "test"
        role[idx[n_test:n_test + n_val]] = "val"
        role[idx[n_test + n_val:]] = "train"
    return role


def split_table(M, seed=0):
    """One row per hold: its role in each cross-signer rotation and in the calibrated-user split."""
    T = M[["occ", "signer", "letter"]].copy()
    sg = M.signer.to_numpy()
    for k, (train, val, test) in enumerate(signer_rotations(sg), 1):
        col = np.array(["unused"] * len(M), dtype=object)   # calibration rows take no part
        col[np.isin(sg, train)] = "train"
        col[sg == val] = "val"
        col[sg == test] = "test"
        T[f"new_person_rotation{k}"] = col
    cal = calibration_split(M, seed)
    cal[~is_user(sg)] = "train"                            # dataset signers: all training
    T["calibrated_user"] = cal
    return T
