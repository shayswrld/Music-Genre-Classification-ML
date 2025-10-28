# Run from inside the fma/ directory:
#   source .venv/bin/activate
#   python baseline_train.py

from pathlib import Path
import json
import numpy as np
from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from joblib import dump

from data_loader import FMADataLoader

# Config
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


METADATA_DIR = str(HERE / "data" / "fma_metadata")
SUBSET       = "large"       
N_SAMPLES    = 1_000_000     # take all available

RESULTS_DIR  = ROOT / "results"
ARTIFACTS    = ROOT / "artifacts"
RESULTS_DIR.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


# Helpers

def pack_metrics(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }

def report(name, y_true, y_pred):
    m = pack_metrics(y_true, y_pred)
    print(f"{name:16s} acc={m['acc']:.4f}  macroF1={m['macro_f1']:.4f}")
    return m


# Load data via loader (authors' split + standardize)

print("Loading FMA (metadata features) with authors' fma-large split…")
loader = FMADataLoader(metadata_path=METADATA_DIR)
data = loader.get_train_test_split(
    n=N_SAMPLES,
    subset=SUBSET,            # fma-large via tracks[('set','subset')]
    feature_columns=None,     # None = all 518 features from features.csv
    multi_label=False,
    include_echonest=False,
    standardize=True,         # z-score on TRAIN; apply to VAL/TEST
    shuffle_data=True
)

Xtr, ytr = data["X_train"], data["y_train"]
Xva, yva = data["X_val"],   data["y_val"]
Xte, yte = data["X_test"],  data["y_test"]
classes  = data.get("classes", None)

print("Loaded splits:",
      "train", len(ytr),
      "val", len(yva),
      "test", len(yte))
print("Shapes:", Xtr.shape, Xva.shape, Xte.shape)


# Trivial baselines (context only)

maj = Counter(ytr).most_common(1)[0][0]
yva_maj = np.full_like(yva, maj)
yte_maj = np.full_like(yte, maj)
maj_val = report("Majority (val)", yva, yva_maj)
maj_tst = pack_metrics(yte, yte_maj)

rng = np.random.default_rng(SEED)
labels, counts = np.unique(ytr, return_counts=True)
probs = counts / counts.sum()
yva_rand = rng.choice(labels, size=len(yva), p=probs)
yte_rand = rng.choice(labels, size=len(yte), p=probs)
rnd_val = report("Stratified (val)", yva, yva_rand)
rnd_tst = pack_metrics(yte, yte_rand)


# Naïve Bayes (for comparison)

gnb = GaussianNB().fit(Xtr, ytr)
gnb_val = report("GaussianNB (val)", yva, gnb.predict(Xva))
gnb_tst = pack_metrics(yte, gnb.predict(Xte))


# k-NN baseline (official simple baseline) — tune k on val

k_grid = [1, 3, 5, 9, 15, 31]
print("\nTuning k for k-NN…")
best_k, best_acc = None, -1.0
best_knn = None
for k in k_grid:
    knn = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2, n_jobs=-1)
    knn.fit(Xtr, ytr)
    acc = accuracy_score(yva, knn.predict(Xva))
    print(f"k={k:<2d}  val acc={acc:.4f}")
    if acc > best_acc:
        best_acc, best_k, best_knn = acc, k, knn

knn = best_knn
knn_val = report(f"kNN(k={best_k}) (val)", yva, knn.predict(Xva))
knn_tst = pack_metrics(yte, knn.predict(Xte))


# Logistic Regression (multinomial, L2) — simple, strong linear baseline

print("\nTuning C for LogisticRegression (multinomial, lbfgs)…")
C_grid = [0.1, 0.3, 1.0, 3.0, 10.0]
best_C, best_acc = None, -1.0
best_lr = None

for C in C_grid:
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    )
    lr.fit(Xtr, ytr)
    acc = accuracy_score(yva, lr.predict(Xva))
    print(f"C={C:<4} val acc={acc:.4f}")
    if acc > best_acc:
        best_acc, best_C, best_lr = acc, C, lr

lr = best_lr
lr_val = report(f"LogReg(C={best_C}) (val)", yva, lr.predict(Xva))
lr_tst = pack_metrics(yte, lr.predict(Xte))


# Save artifacts (models + classes) and metrics JSON

if classes is not None:
    with open(ARTIFACTS / "classes_top16.json", "w") as f:
        json.dump(list(classes), f, indent=2)

dump(gnb, ARTIFACTS / "baseline_gaussiannb.joblib")
dump(knn, ARTIFACTS / f"baseline_knn_k{best_k}.joblib")
dump(lr,  ARTIFACTS / f"baseline_logreg_C{best_C}.joblib")

results = {
    "subset": SUBSET,
    "seed": SEED,
    "shapes": {
        "train": list(Xtr.shape),
        "val":   list(Xva.shape),
        "test":  list(Xte.shape)
    },
    "trivial": {
        "majority_val": maj_val,
        "majority_test": maj_tst,
        "stratified_val": rnd_val,
        "stratified_test": rnd_tst
    },
    "gaussiannb": {
        "val": gnb_val,
        "test": gnb_tst
    },
    "knn": {
        "best_k": best_k,
        "val": knn_val,
        "test": knn_tst
    },
    "logreg": {
        "best_C": best_C,
        "val": lr_val,
        "test": lr_tst
    }
}
with open(RESULTS_DIR / "baseline_metrics_large.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved metrics to {RESULTS_DIR/'baseline_metrics_large.json'}")
print(f"Saved artifacts to {ARTIFACTS}")
print("\nDone.")
