import os
import random
import warnings

import numpy as np
import pandas as pd



from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")


# ===========================
# RANDOM SEED
# ===========================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ===========================
# CONFIGURATION
# ===========================

THRESHOLD = "0.10"

CLASSIFICATION = "binary"
# change to "multiclass" later

N_SPLITS = 10
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001

# ===========================
# DATASET PATH
# ===========================

DATASET_FILE = (
    f"/home/dhanush2026/dhanush2026/AI_IDS/"
    f"PCC_RESULTS/threshold_{THRESHOLD}/reduced_dataset.csv"
)

# ===========================
# LOAD DATASET
# ===========================

print("=" * 60)
print("Loading Reduced Dataset...")
print("=" * 60)

df = pd.read_csv(DATASET_FILE)

print(f"Threshold : {THRESHOLD}")
print(f"Dataset Shape : {df.shape}")
print()

print(df.head())

print("\nColumns\n")
print(df.columns.tolist())

# ==========================================================
# LABEL ENCODING
# ==========================================================

print("\n" + "=" * 60)
print("Label Encoding")
print("=" * 60)

if CLASSIFICATION == "binary":

    # BENIGN -> 0
    # All attacks -> 1
    df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    print("\nBinary Classification Selected")
    print(df["Label"].value_counts())

else:

    encoder = LabelEncoder()
    df["Label"] = encoder.fit_transform(df["Label"])

    print("\nMulti-Class Classification Selected")
    print(f"Number of Classes : {len(encoder.classes_)}")

    print("\nLabel Mapping:\n")

    for i, label in enumerate(encoder.classes_):
        print(f"{i:2d} --> {label}")
        

# ==========================================================
# FEATURES & LABELS
# ==========================================================

X = df.drop(columns=["Label"])

y = df["Label"]

print("\nFeature Matrix Shape :", X.shape)
print("Label Vector Shape   :", y.shape)

# ==========================================================
# CONVERT TO NUMPY
# ==========================================================

X = X.values.astype(np.float32)
y = y.values

print("\nNumPy Conversion Completed")
print("X Shape :", X.shape)
print("y Shape :", y.shape)

# ==========================================================
# STRATIFIED 10-FOLD CROSS VALIDATION
# ==========================================================

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)

print("\n10-Fold Stratified Cross Validation Initialized")
# ==========================================================
# 10-FOLD CROSS VALIDATION
# ==========================================================

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):

    print("\n" + "=" * 70)
    print(f"Fold {fold}/{N_SPLITS}")
    print("=" * 70)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    # Only run one fold for now
    break
    

# ==========================================================
# FEATURE SCALING
# ==========================================================
print("\nChecking for Infinity values...")


feature_names = df.drop(columns=["Label"]).columns

for i, col in enumerate(feature_names):

    if np.isinf(X_train[:, i]).any():

        print(col)
        
print("Positive Inf :", np.isposinf(X_train).sum())

print("Negative Inf :", np.isneginf(X_train).sum())

print("NaN :", np.isnan(X_train).sum())

print("Maximum value :", np.nanmax(X_train))

print("Minimum value :", np.nanmin(X_train))

# ==========================================================
# CLEAN REDUCED DATASET
# ==========================================================

print("\nCleaning Reduced Dataset...")

# Replace Infinity with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Replace NaN with 0
df.fillna(0, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

print("\nChecking Infinity Values...")
print("Positive Inf :", np.isposinf(df.select_dtypes(include=[np.number])).sum().sum())
print("Negative Inf :", np.isneginf(df.select_dtypes(include=[np.number])).sum().sum())

# Save the cleaned dataset
# df.to_csv(DATASET_FILE, index=False) 

#print("\nCleaned dataset saved successfully.")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed")

print("Training Mean :", np.mean(X_train))

print("Training Std  :", np.std(X_train))

# print(df.shape)

# ==========================================================
# RESHAPE FOR CNN
# ==========================================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)

print("\nReshaped Dataset")

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
