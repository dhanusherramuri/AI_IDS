# """
# Project Configuration
# """

# import os

# # ==========================================================
# # RANDOM SEED
# # ==========================================================

# SEED = 42

# # ==========================================================
# # TRAINING PARAMETERS
# # ==========================================================

# N_SPLITS = 10

# BATCH_SIZE = 256

# EPOCHS = 50

# LEARNING_RATE = 0.001

# CLASSIFICATION = "binary"

# # ==========================================================
# # FEATURE SELECTION
# # ==========================================================

# THRESHOLD = "0.10"

# # ==========================================================
# # PROJECT ROOT
# # ==========================================================

# PROJECT_ROOT = "/home/dhanush2026/dhanush2026/AI_IDS"

# # ==========================================================
# # COMMON DATA
# # ==========================================================

# DATASET_ROOT = os.path.join(PROJECT_ROOT, "DATASETS")

# PCC_RESULT_ROOT = os.path.join(PROJECT_ROOT, "PCC_RESULTS")


"""
Project Configuration
"""

import os

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = 42

# ==========================================================
# TRAINING PARAMETERS
# ==========================================================

N_SPLITS = 10

BATCH_SIZE = 256

EPOCHS = 50

LEARNING_RATE = 0.001

CLASSIFICATION = "binary"

# ==========================================================
# FEATURE SELECTION
# ==========================================================

THRESHOLD = "0.10"

# ==========================================================
# PROJECT ROOT
# ==========================================================

import os

_PROJECT_ROOT_LEVELS = 2
_PROJECT_ROOT_BASE = os.path.abspath(__file__)
for _ in range(_PROJECT_ROOT_LEVELS):
    _PROJECT_ROOT_BASE = os.path.dirname(_PROJECT_ROOT_BASE)
PROJECT_ROOT = os.environ.get(
    "AI_IDS_PROJECT_ROOT",
    _PROJECT_ROOT_BASE
)

# ==========================================================
# LINUX PATH COMPATIBILITY
# ==========================================================

# If running on Linux and PROJECT_ROOT doesn't contain 'AI_IDS',
# try to detect based on common Linux directory patterns
if os.name == "posix" and "AI_IDS" not in PROJECT_ROOT:
    # Check if /data/AI_IDS exists (common Linux setup)
    if os.path.isdir("/data/AI_IDS"):
        PROJECT_ROOT = "/data/AI_IDS"
    # Check if /data exists
    elif os.path.isdir("/data"):
        PROJECT_ROOT = "/data"
    # Check if running from /home with project in /home
    elif PROJECT_ROOT.startswith("/home") and os.path.isdir(
        os.path.join(PROJECT_ROOT, "AI_IDS")
    ):
        # Keep original, it already resolved correctly
        pass

# ==========================================================
# COMMON DATA
# ==========================================================

DATASET_ROOT = os.path.join(PROJECT_ROOT, "DATASETS")

PCC_RESULT_ROOT = os.path.join(PROJECT_ROOT, "PCC_RESULTS")
