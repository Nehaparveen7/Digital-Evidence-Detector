import glob
import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from features import extract_forensic_features

# 0: vae_jitter, 1: mean_neutral_spread, 2: edge_crispness,
# 3: grid_ratio, 4: ext_is_png, 13: flat_res_mean
CORE_IDX = [0, 1, 2, 3, 4, 13]

X, y, paths = [], [], []

for filepath in glob.glob("dataset/authentic/**/*.*", recursive=True):
    if os.path.isfile(filepath):
        feat = extract_forensic_features(filepath)
        if feat is not None:
            X.append(feat[CORE_IDX])
            y.append(0)
            paths.append(filepath)

for filepath in glob.glob("dataset/edited/**/*.*", recursive=True):
    if os.path.isfile(filepath):
        feat = extract_forensic_features(filepath)
        if feat is not None:
            X.append(feat[CORE_IDX])
            y.append(1)
            paths.append(filepath)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=2.0, class_weight="balanced", random_state=42)
)
model.fit(X, y)

joblib.dump({"model": model, "indices": CORE_IDX}, "forensic_rf_model.pkl")
print("Saved forensic_rf_model.pkl successfully!\n")

for path, feat_row, label in zip(paths, X, y):
    prob = model.predict_proba([feat_row])[0][1] * 100.0
    print(f"{os.path.basename(path):15s} | Actual: {'EDITED' if label else 'AUTH':6s} | Predicted Edited: {prob:5.1f}%")