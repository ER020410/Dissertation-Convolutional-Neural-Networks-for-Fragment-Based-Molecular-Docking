"""Train default2018 model.ipynb

# Import
"""
from collections import defaultdict
import random
import os, re

import torch, molgrid, gninatorch

import pandas as pd
import matplotlib.pyplot as plt


"""# Training and Test Set"""
infile = "/content/drive/MyDrive/aligned_fragments_refined2/final_dataset.types"
train_file = "/content/train.types"
test_file  = "/content/test.types"

# Group lines by fragment ID
frag2lines = defaultdict(list)

with open(infile) as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split()
        ligand_path = parts[4]  # format: <label> <flexlabel> <rmsd> <rec_path> <lig_path>
        frag_id = os.path.basename(ligand_path).split("_pose")[0]  # assumes 'fragID_poseX.sdf'
        frag2lines[frag_id].append(line)

# Shuffle fragments
random.seed(42)
frag_ids = list(frag2lines.keys())
random.shuffle(frag_ids)

#  Split fragment groups
n = len(frag_ids)
n_train = int(0.8 * n)
train_ids = frag_ids[:n_train]
test_ids  = frag_ids[n_train:]

train_lines = [l for fid in train_ids for l in frag2lines[fid]]
test_lines  = [l for fid in test_ids for l in frag2lines[fid]]

random.seed(42)
random.shuffle(train_lines)
random.shuffle(test_lines)

with open(train_file, "w") as f:
    f.writelines(train_lines)
with open(test_file, "w") as f:
    f.writelines(test_lines)

print(f"Done. Fragments: Train={len(train_ids)}, Test={len(test_ids)} | Lines: Train={len(train_lines)}, Test={len(test_lines)}")

"""# Choice of Hyperparameters

## 500 iterations - Model A (0.001 & 0.001)
"""

!python /content/training_finetune.py \
  /content/train.types \
  --testfile /content/test.types \
  -d /content/drive/MyDrive/aligned_fragments_refined2 \
  --model default2018 \
  --pretrained_model /content/crossdock_default2018.pt \
  --balanced \
  --iterations 500 \
  --batch_size 64 \
  --base_lr 0.001 \
  --weight_decay 0.001 \
  -t 100 \
  --progress_bar \
  --out_dir ./crossdock_2018_0blocks_500_0 \
  --gpu cuda:0

"""## 500 iterations - Model A (5e-4 & 1e-4)"""

!python /content/training_finetune.py \
  /content/train.types \
  --testfile /content/test.types \
  -d /content/drive/MyDrive/aligned_fragments_refined2 \
  --model default2018 \
  --pretrained_model /content/crossdock_default2018.pt \
  --balanced \
  --iterations 500 \
  --batch_size 64 \
  --base_lr 5e-4 \
  --weight_decay 1e-4 \
  -t 100 \
  --progress_bar \
  --out_dir ./crossdock_2018_0blocks_500_1 \
  --gpu cuda:0

"""## 500 iterations - Model A (1e-4 & 1e-4)"""

!python /content/training_finetune.py \
  /content/train.types \
  --testfile /content/test.types \
  -d /content/drive/MyDrive/aligned_fragments_refined2 \
  --model default2018 \
  --pretrained_model /content/crossdock_default2018.pt \
  --balanced \
  --iterations 500 \
  --batch_size 64 \
  --base_lr 1e-4 \
  --weight_decay 1e-4 \
  -t 100 \
  --progress_bar \
  --out_dir ./crossdock_2018_0blocks_500_2 \
  --gpu cuda:0

"""# Model A 1000 Iterations"""

!python /content/training_finetune.py \
  /content/train.types \
  --testfile /content/test.types \
  -d /content/drive/MyDrive/aligned_fragments_refined2 \
  --model default2018 \
  --pretrained_model /content/crossdock_default2018.pt \
  --balanced \
  --iterations 1000 \
  --batch_size 64 \
  --base_lr 1e-4 \
  --weight_decay 1e-4 \
  -t 100 \
  --progress_bar \
  --out_dir ./crossdock_2018_0blocks_1000_1e4 \
  --gpu cuda:0

"""# Model B 1000 Iterations"""

!python /content/training_finetune.py \
  /content/train.types \
  --testfile /content/test.types \
  -d /content/drive/MyDrive/aligned_fragments_refined2 \
  --model default2018 \
  --pretrained_model /content/crossdock_default2018.pt \
  --freeze_prefixes features.unit1 features.unit2 \
  --balanced \
  --iterations 1000 \
  --batch_size 64 \
  --base_lr 1e-4 \
  --weight_decay 1e-4 \
  -t 100 \
  --progress_bar \
  --out_dir ./crossdock_2018_2blocks_1000_1e4 \
  --gpu cuda:0

"""# Model C 1000 Iterations"""

!python /content/training_finetune.py \
  /content/train.types \
  --testfile /content/test.types \
  -d /content/drive/MyDrive/aligned_fragments_refined2 \
  --model default2018 \
  --pretrained_model /content/crossdock_default2018.pt \
  --freeze_prefixes features.unit1 features.unit2 features.unit3 features.unit4 \
  --balanced \
  --iterations 1000 \
  --batch_size 64 \
  --base_lr 1e-4 \
  --weight_decay 1e-4 \
  -t 100 \
  --progress_bar \
  --out_dir ./crossdock_2018_4blocks_1000_1e4 \
  --gpu cuda:0

"""

# Plot the train and validation Pose Loss and ROC-AUC Graph
"""

# Path
log_paths = {
    "Model A (Full Finetuning)": "/content/training model a 1000.log",
    "Model B (Freeze Unit 1-2)": "/content/training model b 1000.log",
    "Model C (Freeze Unit 1-4)": "/content/training model c 1000.log",
}

def parse_gnina_log(filepath):
    data = {
        "epoch_train": [], "train_loss": [], "train_auc": [],
        "epoch_test": [],  "test_loss": [],  "test_auc": []
    }
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_epoch = None
    mode = None
    for line in lines:
        if ">>> Train Results - Epoch[" in line:
            current_epoch = int(re.search(r"Epoch\[(\d+)\]", line).group(1))
            mode = "train"
        elif ">>> Test Results - Epoch[" in line:
            current_epoch = int(re.search(r"Epoch\[(\d+)\]", line).group(1))
            mode = "test"
        elif "Pose Loss:" in line:
            loss = float(line.strip().split(":")[1])
            if mode == "train":
                data["epoch_train"].append(current_epoch)
                data["train_loss"].append(loss)
            elif mode == "test":
                data["epoch_test"].append(current_epoch)
                data["test_loss"].append(loss)
        elif "ROC AUC:" in line:
            auc = float(line.strip().split(":")[1])
            if mode == "train":
                data["train_auc"].append(auc)
            elif mode == "test":
                data["test_auc"].append(auc)
    return data

all_logs = {name: parse_gnina_log(path) for name, path in log_paths.items()}

# assign different colour for different models
colors = {
    "Model A (Full Finetuning)": "tab:blue",
    "Model B (Freeze Unit 1-2)": "tab:orange",
    "Model C (Freeze Unit 1-4)": "tab:green",
}

# Loss Plot
plt.figure(figsize=(9, 5.5))
for name, log in all_logs.items():
    c = colors[name]
    plt.plot(log["epoch_train"], log["train_loss"], label=f"{name} - Train", color=c, linestyle="--", linewidth=1.8)
    plt.plot(log["epoch_test"],  log["test_loss"],  label=f"{name} - Validation",  color=c, linestyle="-", linewidth=1.8)
plt.xlabel("Training Step", fontsize=16)
plt.ylabel("Pose Classification Loss", fontsize=16)
# plt.title("Train and Test Pose Classification Loss Across Training Steps", fontsize=18)
plt.legend(fontsize=10)
plt.tick_params(labelsize=12)
plt.tight_layout()

# ROC-AUC Plot
plt.figure(figsize=(9, 5.5))
for name, log in all_logs.items():
    c = colors[name]
    plt.plot(log["epoch_train"], log["train_auc"], label=f"{name} - Train", color=c, linestyle="--", linewidth=1.8)
    plt.plot(log["epoch_test"],  log["test_auc"],  label=f"{name} - Validation",  color=c, linestyle="-", linewidth=1.8)
plt.xlabel("Training Step", fontsize=16)
plt.ylabel("ROC-AUC", fontsize=16)
# plt.title("Train and Test ROC-AUC Across Training Steps", fontsize=18)
plt.legend(fontsize=10)
plt.tick_params(labelsize=12)
plt.tight_layout()
