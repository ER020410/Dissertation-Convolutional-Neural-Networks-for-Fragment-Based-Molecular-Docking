"""Benchmark for LEADS_FRAGS_no_overlap
"""
"""# Benchmark for LEAD_FRAG_no_overlap dataset

## Import Libraries
"""
# from google.colab import drive
# drive.mount('/content/drive')

import os
import re
import shutil
import AutoDockTools
from glob import glob
import subprocess

import molgrid
import torch
from default2018_model import default2018_Net
from meeko import MoleculePreparation
from rdkit import Chem
from tqdm import tqdm

from openbabel import openbabel
openbabel.OBMessageHandler().SetOutputLevel(0)
openbabel.obErrorLog.SetOutputLevel(0)

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog('rdApp.*')

import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from rmsd_utils import robust_rmsd

from sklearn.metrics import roc_curve, auc
"""

## Check train/test Overlap
"""
# .types path
types_file_path = '/content/drive/MyDrive/ref_uff_train0.types'

# LEADS_FRAGS
leads_frag_root = '/content/drive/MyDrive/LEADS-FRAGS'

def extract_two_ids_from_path(path):
    filename = path.split('/')[-1]  
    parts = filename.split('_')

    if len(parts) >= 4:
        return [parts[0], parts[3]]  # first and third are PDB_ID
    elif len(parts) >= 1:
        return [parts[0]]
    else:
        return []

types_sample_ids = set()
with open(types_file_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        path = parts[4]
        ids = extract_two_ids_from_path(path)
        for id_ in ids:
            types_sample_ids.add(id_)

print(f"Extracted {len(types_sample_ids)} unique PDB IDs from types file.")

types_sample_ids_upper = set(id_.upper() for id_ in types_sample_ids)
df_types = pd.DataFrame(sorted(types_sample_ids_upper), columns=['sample_id'])

leads_sample_ids = set(os.listdir(leads_frag_root))
leads_sample_ids_upper = set(id_.upper() for id_ in leads_sample_ids)
df_leads = pd.DataFrame(sorted(leads_sample_ids_upper), columns=['sample_id'])

overlap = types_sample_ids_upper.intersection(leads_sample_ids_upper)
print(f'Number of overlapping sample_ids: {len(overlap)}')

leads_frag_root = './LEADS-FRAGS'
new_dir = './LEADS_FRAGS_no_overlap'
os.makedirs(new_dir, exist_ok=True)

overlap_upper = set(id_.upper() for id_ in overlap)

for item in os.listdir(leads_frag_root):
    item_upper = item.upper()
    src_path = os.path.join(leads_frag_root, item)
    dst_path = os.path.join(new_dir, item)
    if item_upper not in overlap_upper:
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
print(f"Copied non-overlapping items to {new_dir}")
"""

## Data processing
"""

input_root = './LEADS_FRAGS_no_overlap'

ligand_files = glob(os.path.join(input_root, '*', '*_ligand.mol2'))
receptor_files = glob(os.path.join(input_root, '*', '*_receptor.mol2'))

print(f'Found {len(ligand_files)} ligand mol2 files.')
print(f'Found {len(receptor_files)} receptor mol2 files.')

# Optional: print first 5 entries for inspection
print("\nSample ligand files:")
for path in ligand_files[:5]:
    print(path)

print("\nSample receptor files:")
for path in receptor_files[:5]:
    print(path)

"""## Run Models"""

# Fix seeds
seed=42
molgrid.set_random_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Set CuDNN options for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Helper function to get predictions and labels
def get_predictions_gnina_pose(model, test_file, label_idx=0, pred_idx=0, batch_size=20, data_root='./'):
    ypred_test, y_test = [], []
    model.eval()
    with torch.no_grad():
        e_test = molgrid.ExampleProvider(data_root=data_root,balanced=False,shuffle=False)
        e_test.populate(test_file)
        gmaker = molgrid.GridMaker()
        dims = gmaker.grid_dimensions(e_test.num_types())
        tensor_shape = (batch_size,)+dims
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
        float_labels = torch.zeros(batch_size, dtype=torch.float32)

        num_samples = e_test.size()
        num_batches = -(-num_samples // batch_size)
        for _ in range(num_batches):
            # Load data
            batch = e_test.next_batch(batch_size)
            batch.extract_label(label_idx, float_labels)
            gmaker.forward(batch, input_tensor, random_rotation=False, random_translation=0.0)
            # Get prediction
            output = model(input_tensor)[pred_idx].detach().cpu().numpy().reshape(-1)
            ypred_test.extend(list(output))
            # Get labels
            y_test.extend(list(float_labels.detach().cpu().numpy()))
    ypred_test = np.array(ypred_test)[:num_samples]
    y_test = np.array(y_test)[:num_samples]
    return ypred_test, y_test

data_root = './LEADS_FRAGS_no_overlap'
types_file = os.path.join(data_root, 'leads_frags_prepared.types')

ligand_files = glob(os.path.join(data_root, '*', '*_ligand.mol2'))

with open(types_file, 'w') as f:
    for ligand_path in ligand_files:
        pdbid = os.path.basename(ligand_path).split('_')[0]
        receptor_rel = f'{pdbid}/{pdbid}_receptor.mol2'
        ligand_rel = f'{pdbid}/{pdbid}_ligand.mol2'
        label = 0.0 # not sure how to find affinity labels
        f.write(f'{label} {receptor_rel} {ligand_rel}\n')

print(f" .types created: {types_file}")

"""###  Default2018 - CrossDocked:Predictive performance - LEADS_FRAGS_no_overlap

"""

data_name = 'LEADS_FRAGS_no_overlap'
data_root = './LEADS_FRAGS_no_overlap'

model_name = './crossdock_default2018.pt'
dims = (28, 48, 48, 48)
model = default2018_Net(dims).to('cuda')
model.load_state_dict(torch.load(model_name))

preds, labels = get_predictions_gnina_pose(model, types_file, data_root=data_root)

rmse = np.sqrt(np.mean((labels-preds)**2))
corr = pearsonr(preds, labels)[0]
print(f'Performance default2018 on {data_name} - RMSE: {rmse:.3f}, Pearson: {corr:.3f}')

"""# Redocking and Rescoring"""

docking_output_dir = '/content/drive/MyDrive/LEADS_FRAGS_no_overlap_docking_results'
os.makedirs(docking_output_dir, exist_ok=True)

seed = 42
autobox_add = 24
batch_size = 10

def batch_redocking_with_logs_in_batches(data_root, docking_output_dir, batch_size=10, seed=42, autobox_add=24):
    complex_dirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    batch_size
    for i in range(0, len(complex_dirs), batch_size):
        batch = complex_dirs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}: samples {i+1} to {i+len(batch)}")
        for idx in batch:
            receptor_path = os.path.join(data_root, idx, f'{idx}_receptor.mol2')
            ligand_path = os.path.join(data_root, idx, f'{idx}_ligand.mol2')
            ref_ligand_path = os.path.join(data_root, idx, f'{idx}_ref-ligand.pdb')

            if not (os.path.exists(receptor_path) and os.path.exists(ligand_path) and os.path.exists(ref_ligand_path)):
                print(f'Missing files for {idx}, skipping.')
                continue

            output_path = os.path.join(docking_output_dir, f'{idx}_docked.sdf')
            log_path = os.path.join(docking_output_dir, f'{idx}_docking.log')

            cmd = [
                './smina.static',
                '-r', receptor_path,
                '-l', ligand_path,
                '--autobox_ligand', ref_ligand_path,
                '--autobox_add', str(autobox_add),
                '-o', output_path,
                '--exhaustiveness', '64',
                '--num_modes', '40',
                '--seed', str(seed),
                '--log', log_path
            ]

            print(f'Running docking for {idx}...')
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f'Error docking {idx}:', result.stderr.decode())
            else:
                print(f'Docking finished for {idx}, result saved at {output_path}, log saved at {log_path}')
        print(f"Finished batch {i//batch_size + 1}\n")

if __name__ == '__main__':
    batch_redocking_with_logs_in_batches(data_root, docking_output_dir, batch_size=batch_size, seed=seed, autobox_add=autobox_add)

docking_root = './LEADS_FRAGS_no_overlap_docking_results'

complex_ids = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

all_rmsds = {}

for idx in complex_ids:
    ref_path = os.path.join(data_root, idx, f'{idx}_ref-ligand.pdb')
    docked_path = os.path.join(docking_root, f'{idx}_docked.sdf')

    if not (os.path.exists(ref_path) and os.path.exists(docked_path)):
        print(f"Missing files for {idx}, skipping...")
        continue

    ref_mol = Chem.MolFromPDBFile(ref_path, removeHs=False)
    docked_mols = Chem.SDMolSupplier(docked_path)

    rmsds = []
    for mol in docked_mols:
        if mol is None:
            continue
        rmsds.append(robust_rmsd(mol, ref_mol))

    all_rmsds[idx] = rmsds

    print(f"Sample {idx} RMSDs: {rmsds}")

rows = []

for idx in complex_ids:
    ref_path = os.path.join(data_root, idx, f'{idx}_ref-ligand.pdb')
    docked_path = os.path.join(docking_root, f'{idx}_docked.sdf')

    if not (os.path.exists(ref_path) and os.path.exists(docked_path)):
        print(f"Missing files for {idx}, skipping...")
        continue

    ref_mol = Chem.MolFromPDBFile(ref_path, removeHs=False)
    docked_mols = Chem.SDMolSupplier(docked_path)

    for rank, mol in enumerate(docked_mols, start=1):
        if mol is None:
            continue
        rmsd_value = robust_rmsd(mol, ref_mol)
        rows.append({'pose_rank': rank, 'sample_id': idx, 'rmsd': rmsd_value})

RMSDs_LEADS_FRAGS_no_overlap = pd.DataFrame(rows, columns=['pose_rank', 'sample_id', 'rmsd'])
RMSDs_LEADS_FRAGS_no_overlap.to_csv('RMSDs_LEADS_FRAGS_no_overlap.csv', index=False)
print('Saved predictions to RMSDs_LEADS_FRAGS_no_overlap.csv')

def extract_affinity_from_log(log_path):
    affinities = []
    with open(log_path, 'r') as f:
        lines = f.readlines()

    reading = False
    for line in lines:
        if 'mode |' in line:
            reading = True
            continue
        if reading:
            if line.startswith('-----') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                break
            try:
                pose_rank = int(parts[0])
                affinity = float(parts[1])
                affinities.append(affinity)
            except ValueError:
                continue

    return affinities

def batch_extract_affinity(log_dir, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['pose_rank', 'sample_id', 'affinity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(log_dir):
            if filename.endswith('.log'):
                filepath = os.path.join(log_dir, filename)
                affinities = extract_affinity_from_log(filepath)
                sample_id = filename.split('_')[0]
                if affinities:
                    for i, affinity in enumerate(affinities, 1):
                        writer.writerow({
                            'pose_rank': i,
                            'sample_id': sample_id,
                            'affinity': affinity
                        })
                else:
                    print(f"No affinity scores found in {filename}")

    print(f"Batch extraction complete. Results saved to {output_csv}")

log_folder = './LEADS_FRAGS_no_overlap_docking_results'
output_file = 'smina_affinity_scores.csv'

batch_extract_affinity(log_folder, output_file)

smina_affinity_scores = './smina_affinity_scores.csv'
smina_affinity_scores = pd.read_csv(smina_affinity_scores)

df_merged = pd.merge(smina_affinity_scores, RMSDs_LEADS_FRAGS_no_overlap, on=['sample_id', 'pose_rank'], how='inner')

def assign_label(df, rmsd_threshold=2.0):
    df['label'] = (df['rmsd'] <= rmsd_threshold).astype(int)
    return df

smina_prediction_no_overlap = assign_label(df_merged)

smina_prediction_no_overlap.to_csv('smina_prediction_no_overlap.csv', index=False)
print("Label assignment done. Saved to smina_prediction_no_overlap.csv")

smina_prediction_no_overlap = './smina_prediction_LEADS_FRAGS_no_overlap.csv'
smina_prediction_no_overlap = pd.read_csv(smina_prediction_no_overlap)

"""# Rerank using gnina"""

# Fix seeds
seed = 42 # Define seed here
molgrid.set_random_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Set CuDNN options for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dims = (28, 48, 48, 48)
model_name = './crossdock_default2018.pt'
model = default2018_Net(dims).to('cuda')
model.load_state_dict(torch.load(model_name))
model.eval()

data_root = './LEADS_FRAGS_no_overlap'
docking_root = './LEADS_FRAGS_no_overlap_docking_results'

complex_ids = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

# .types file
for idx in complex_ids:
    docked_sdf_path = os.path.join(docking_root, f'{idx}_docked.sdf')
    receptor_path = os.path.join(data_root, idx, f'{idx}_receptor.mol2')
    ref_path = os.path.join(data_root, idx, f'{idx}_ref-ligand.pdb')

    if not (os.path.exists(docked_sdf_path) and os.path.exists(receptor_path) and os.path.exists(ref_path)):
        print(f"Missing files for {idx}, skipping...")
        continue

    ref_mol = Chem.MolFromPDBFile(ref_path, removeHs=False)
    docked_mols = Chem.SDMolSupplier(docked_sdf_path)

    rmsds = []
    for mol in docked_mols:
        if mol is None:
            continue
        rmsds.append(robust_rmsd(mol, ref_mol))

    pose_dir = os.path.join(data_root, idx, 'poses')
    os.makedirs(pose_dir, exist_ok=True)

    types_file_path = os.path.join(data_root, idx, f'{idx}.types')

    with open(types_file_path, 'w') as f:
        for i, (mol, rmsd) in enumerate(zip(docked_mols, rmsds)):
            if mol is None:
                continue
            label = int(rmsd <= 2)

            pose_sdf_path = os.path.join(pose_dir, f'{idx}_docked_{i}.sdf')
            with Chem.SDWriter(pose_sdf_path) as writer:
                writer.write(mol)

            rel_receptor_path = os.path.relpath(receptor_path, data_root)
            rel_pose_sdf_path = os.path.relpath(pose_sdf_path, data_root)

            f.write(f'{label} {rmsd:.2f} {rel_receptor_path} {rel_pose_sdf_path}\n')

    print(f"Generated .types for {idx} with {len(rmsds)} poses.")

# Predict pose scores
def get_predictions_pose(model, test_file, label_idx=0, pred_idx=0, batch_size=20, data_root='./'):
    ypred_test, y_test = [], []
    model.eval()
    with torch.no_grad():
        e_test = molgrid.ExampleProvider(data_root=data_root, balanced=False, shuffle=False)
        e_test.populate(test_file)
        gmaker = molgrid.GridMaker()
        dims = gmaker.grid_dimensions(e_test.num_types())
        tensor_shape = (batch_size,) + dims
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
        float_labels = torch.zeros(batch_size, dtype=torch.float32)

        num_samples = e_test.size()
        num_batches = -(-num_samples // batch_size)
        for _ in range(num_batches):
            batch = e_test.next_batch(batch_size)
            batch.extract_label(label_idx, float_labels)
            gmaker.forward(batch, input_tensor, random_rotation=False, random_translation=0.0)
            output = model(input_tensor)[pred_idx].detach().cpu().numpy().reshape(-1)
            ypred_test.extend(list(output))
            y_test.extend(list(float_labels.detach().cpu().numpy()))
    ypred_test = np.array(ypred_test)[:num_samples]
    y_test = np.array(y_test)[:num_samples]
    return ypred_test, y_test

# Predict affinity scores
def get_predictions_affinity(model, test_file, label_idx=0, pred_idx=1, batch_size=20, data_root='./'):
    ypred_test, y_test = [], []
    model.eval()
    with torch.no_grad():
        e_test = molgrid.ExampleProvider(data_root=data_root, balanced=False, shuffle=False)
        e_test.populate(test_file)
        gmaker = molgrid.GridMaker()
        dims = gmaker.grid_dimensions(e_test.num_types())
        tensor_shape = (batch_size,) + dims
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
        float_labels = torch.zeros(batch_size, dtype=torch.float32)

        num_samples = e_test.size()
        num_batches = -(-num_samples // batch_size)
        for _ in range(num_batches):
            batch = e_test.next_batch(batch_size)
            batch.extract_label(label_idx, float_labels)
            gmaker.forward(batch, input_tensor, random_rotation=False, random_translation=0.0)
            output = model(input_tensor)[pred_idx].detach().cpu().numpy().reshape(-1)
            ypred_test.extend(list(output))
            y_test.extend(list(float_labels.detach().cpu().numpy()))
    ypred_test = np.array(ypred_test)[:num_samples]
    y_test = np.array(y_test)[:num_samples]
    return ypred_test, y_test

all_sample_ids = []
all_pose_preds = []
all_affinity_preds = []
all_labels = []

for idx in complex_ids:
    types_path = os.path.join(data_root, idx, f'{idx}.types')
    if not os.path.exists(types_path):
        print(f"Missing .types file for {idx}, skipping...")
        continue

    pose_preds, labels = get_predictions_pose(model, types_path, label_idx=0, pred_idx=0, batch_size=20, data_root=data_root)
    affinity_preds, _ = get_predictions_affinity(model, types_path, label_idx=0, pred_idx=1, batch_size=20, data_root=data_root)

    if len(pose_preds) == 0:
        print(f"No samples loaded for {idx}")
        continue

    all_sample_ids.extend([idx] * len(pose_preds))
    all_pose_preds.extend(pose_preds)
    all_affinity_preds.extend(affinity_preds)
    all_labels.extend(labels)

 # Summarize in DataFrame
df = pd.DataFrame({
    'sample_id': all_sample_ids,
    'pose_prediction': all_pose_preds,
    'affinity_prediction': all_affinity_preds,
    'label': all_labels
})

df.to_csv('gnina_predictions_LEADS_FRAGS_no_overlap.csv', index=False)
print("Saved predictions to gnina_predictions_LEADS_FRAGS_no_overlap.csv")

gnina_predictions = './gnina_predictions_LEADS_FRAGS_no_overlap.csv'
gnina_predictions = pd.read_csv(gnina_predictions)

smina_predictions = './smina_prediction_LEADS_FRAGS_no_overlap.csv'
smina_predictions = pd.read_csv(smina_predictions)

# Distribution of Scores
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

# SMINA Affinity
smina_scores = smina_predictions["affinity"]
sns.histplot(smina_scores, bins=40, kde=True, ax=axes[0], color="seagreen")
axes[0].set_title("Distribution of SMINA Affinity Scores", fontsize=18)
axes[0].set_ylim(0, 250)
axes[0].set_xlabel("Score", fontsize=16)
axes[0].set_ylabel("")
axes[0].tick_params(axis='both', which='major', labelsize=12)

# GNINA Affinity
affinity_scores = gnina_predictions["affinity_prediction"]
sns.histplot(affinity_scores, bins=40, kde=True, ax=axes[1], color="steelblue")
axes[1].set_title("Distribution of GNINA Affinity Scores", fontsize=18)
axes[1].set_ylim(0, 250)
axes[1].set_xlabel("Score", fontsize=16)
axes[1].set_ylabel("Frequency", fontsize=16)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# GNINA Pose
pose_scores = gnina_predictions["pose_prediction"]
sns.histplot(pose_scores, bins=40, kde=True, ax=axes[2], color="darkorange")
axes[2].set_title("Distribution of GNINA Pose Scores", fontsize=18)
axes[2].set_ylim(0, 250)
axes[2].set_xlabel("Score", fontsize=16)
axes[2].set_ylabel("")
axes[2].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()

# Hybrid Metrics
sample_ids = gnina_predictions.iloc[:, 0].values
pose_scores = gnina_predictions.iloc[:, 1].values
affinity_scores = gnina_predictions.iloc[:, 2].values
label = gnina_predictions.iloc[:, 3].values

w1, w2 = 0.6, 0.4
combined_mul = pose_scores * affinity_scores
combined_weighted_sum = w1 * pose_scores + w2 * affinity_scores
combined_geom_mean = np.sqrt(pose_scores * affinity_scores)
combined_log_sum = np.exp(w1 * np.log(pose_scores + 1e-8) + w2 * np.log(pose_scores + 1e-8))
alpha = 0.1
combined_pow = np.power(pose_scores, alpha) * np.power(affinity_scores, 1 - alpha)

df_combined_no_overlap = pd.DataFrame({
    'sample_id': sample_ids,
    'pose_score': pose_scores,
    'affinity_score': affinity_scores,
    'multiplication': combined_mul,
    'weighted_sum': combined_weighted_sum,
    'geom_mean': combined_geom_mean,
    'log_sum': combined_log_sum,
    'power': combined_pow,
    'label': label
})

# save
df_combined_no_overlap.to_csv('df_combined_no_overlap_extended_all.csv', index=False)

score_columns = [
    col for col in df_combined_no_overlap.columns
    if df_combined_no_overlap[col].dtype in [np.float64, np.int64]
    and col != 'sample_id'
]

corr_matrix = df_combined_no_overlap[score_columns].corr()

# heatmap
plt.figure(figsize=(max(10, len(score_columns)), max(8, len(score_columns) * 0.5)))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    annot_kws={"size": 14},
    xticklabels=score_columns,
    yticklabels=score_columns
)

plt.xticks(fontsize=16, rotation=45, ha='right')
plt.yticks(fontsize=16)
plt.title("Correlation Matrix of Scoring Metrics", fontsize=20)
plt.tight_layout()
plt.show()

"""## Succuss Pose Rank"""

def get_predictions_gnina(model, test_file, label_idx=1, pred_idx=0, batch_size=32, data_root='./'):
    ypred_test, y_test = [], []
    model.eval()
    with torch.no_grad():
        e_test = molgrid.ExampleProvider(data_root=data_root,balanced=False,shuffle=False)
        e_test.populate(test_file)
        gmaker = molgrid.GridMaker()
        dims = gmaker.grid_dimensions(e_test.num_types())
        tensor_shape = (batch_size,)+dims
        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda')
        float_labels = torch.zeros(batch_size, dtype=torch.float32)

        num_samples = e_test.size()
        num_batches = -(-num_samples // batch_size)
        for _ in range(num_batches):
            batch = e_test.next_batch(batch_size)
            batch.extract_label(label_idx, float_labels)
            gmaker.forward(batch, input_tensor, random_rotation=False, random_translation=0.0)
            # Get prediction
            output = model(input_tensor)[pred_idx].detach().cpu().numpy().reshape(-1)
            ypred_test.extend(list(output))
            # Get labels
            y_test.extend(list(float_labels.detach().cpu().numpy()))
    ypred_test = np.array(ypred_test)[:num_samples]
    y_test = np.array(y_test)[:num_samples]
    return ypred_test, y_test

def assess_first_success(rmsds, preds, labels=None):
    rmsds = np.array(rmsds)
    preds = np.array(preds)

    success_possible = np.any(rmsds <= 2)
    if not success_possible:
        return None, None

    smina_first_success_rank = np.argmax(rmsds <= 2) + 1

    sorted_indices = np.argsort(-preds)
    sorted_rmsds = rmsds[sorted_indices]
    gnina_first_success_rank = np.argmax(sorted_rmsds <= 2) + 1

    return smina_first_success_rank, gnina_first_success_rank

all_rmsds = {}
all_preds = {}
all_labels = {}

for idx in complex_ids:
    ref_path = os.path.join(data_root, idx, f'{idx}_ref-ligand.pdb')
    docked_path = os.path.join(docking_root, f'{idx}_docked.sdf')

    if not (os.path.exists(ref_path) and os.path.exists(docked_path)):
        print(f"Missing files for {idx}, skipping...")
        continue

    # RMSD
    ref_mol = Chem.MolFromPDBFile(ref_path, removeHs=False)
    docked_mols = Chem.SDMolSupplier(docked_path)
    rmsds = []
    for mol in docked_mols:
        if mol is None:
            continue
        rmsds.append(robust_rmsd(mol, ref_mol))
    all_rmsds[idx] = rmsds
    print(f"Sample {idx} RMSDs: {rmsds}")

    # pose score
    types_path = os.path.join(data_root, idx, f'{idx}.types')
    if not os.path.exists(types_path):
        print(f"Missing .types file for {idx}, skipping prediction...")
        continue

    preds, labels = get_predictions_gnina(
        model,
        test_file=types_path,
        label_idx=0,  # pose true label(0/1)
        pred_idx=0,   # pose score prediction
        batch_size=32,
        data_root=data_root
    )

    if len(preds) == 0:
        print(f"No predictions for {idx}")
        continue

    all_preds[idx] = preds
    all_labels[idx] = labels

# evaluation
for idx in all_rmsds.keys():
    smina_rank, gnina_rank = assess_first_success(all_rmsds[idx], all_preds.get(idx, []))
    if smina_rank is None:
        print(f"{idx}: No successful pose (RMSD ≤ 2)")
    else:
        print(f"{idx}: smina first success pose rank: {smina_rank}, gnina first success pose rank: {gnina_rank}")

results = []

for idx in all_rmsds.keys():
    smina_rank, gnina_rank = assess_first_success(all_rmsds[idx], all_preds.get(idx, []))
    if smina_rank is None:
        line = f"{idx}: No successful pose (RMSD ≤ 2)"
    else:
        line = f"{idx}: smina first success pose rank: {smina_rank}, gnina first success pose rank: {gnina_rank}"
    results.append(line)

with open('docking_success_results_no_overlap.txt', 'w') as f:
    f.write('\n'.join(results))

print("Saved results to docking_success_results_no_overlap.txt")

input_txt = './docking_success_results_no_overlap.txt'
output_csv = 'success_pose_ranks_no_overlap.csv'

rows = []

with open(input_txt, 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()
    if not line:
        continue
    idx = i + 1
    if 'No successful pose' in line:
        sample_id = line.split(':')[0].strip()
        smina_rank = 0
        gnina_rank = 0
    else:
        sample_id_match = re.match(r'^(\S+):', line)
        sample_id = sample_id_match.group(1) if sample_id_match else ''
        smina_rank_match = re.search(r'smina first success pose rank: (\d+)', line)
        gnina_rank_match = re.search(r'gnina first success pose rank: (\d+)', line)
        smina_rank = int(smina_rank_match.group(1)) if smina_rank_match else 0
        gnina_rank = int(gnina_rank_match.group(1)) if gnina_rank_match else 0

    rows.append({
        'idx': idx,
        'sample_id': sample_id,
        'smina_first_success': smina_rank,
        'gnina_first_success': gnina_rank
    })

with open(output_csv, 'w', newline='') as csvfile:
    fieldnames = ['idx', 'sample_id', 'smina_first_success', 'gnina_first_success']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {output_csv}")

# Rank Analysis
pose_rank = './success_pose_ranks_no_overlap.csv'
pose_rank = pd.read_csv(pose_rank)

pose_rank_no_zero = pose_rank[(pose_rank != 0).all(axis=1)].reset_index(drop=True)
print(f"Original: {pose_rank.shape[0]} rows, Filtered: {pose_rank_no_zero.shape[0]} rows")

# Average pose rank of 1st successful docking for smina and gnina
smina_first_success = pose_rank['smina_first_success'].values
gnina_first_success = pose_rank['gnina_first_success'].values

ave_smina = np.mean(smina_first_success)
ave_gnina = np.mean(gnina_first_success)

print('Average pose rank of 1st successful docking for smina:', ave_smina)
print('Average pose rank of 1st successful docking for gnina:', ave_gnina)

# Top-N pose rank
def count_success_top_n(pose_rank, rank_col, n=1):
    # only count the success rank equal to n
    success = (pose_rank[rank_col] <= n).sum()
    return success

for n in [1, 3, 5, 10, 15, 20, 25, 30, 34, 35, 36, 37, 38, 39, 40]:
    print(f'Top {n} success count (smina): {count_success_top_n(pose_rank_no_zero, "smina_first_success", n)}')
    print(f'Top {n} success count (gnina): {count_success_top_n(pose_rank_no_zero, "gnina_first_success", n)}')

pose_rank = '/content/drive/MyDrive/success_pose_ranks_no_overlap.csv'
pose_rank = pd.read_csv(pose_rank)
pose_rank_no_zero = pose_rank[(pose_rank != 0).all(axis=1)].reset_index(drop=True)

# visualize
top_n_range = list(range(1, 40))
success_rates_smina = []
success_rates_gnina = []

for n in top_n_range:
    success_smina = count_success_top_n(pose_rank_no_zero, 'smina_first_success', n=n)
    success_gnina = count_success_top_n(pose_rank_no_zero, 'gnina_first_success', n=n)

    total_complexes = 59
    success_rates_smina.append(success_smina / total_complexes)
    success_rates_gnina.append(success_gnina / total_complexes)

plt.figure(figsize=(10, 6))

plt.plot(top_n_range, success_rates_smina, label='SMINA', marker='o')
plt.plot(top_n_range, success_rates_gnina, label='GNINA', marker='s')
plt.xlabel('Pose Rank Threshold (N)', fontsize=20)
plt.ylabel('Success Rate', fontsize=20)
plt.legend(fontsize=18, loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.tick_params(labelsize=16)

plt.tight_layout()
plt.show()

"""### Distribution of success pose rank"""

# rank values
smina_sorted = np.sort(pose_rank_no_zero['smina_first_success'].values)
gnina_sorted = np.sort(pose_rank_no_zero['gnina_first_success'].values)
overlap_mask = (smina_sorted == gnina_sorted)

plt.figure(figsize=(12, 6))

plt.scatter(np.where(overlap_mask)[0], smina_sorted[overlap_mask], color='green', label='Overlap', s=70, alpha=0.8)
plt.scatter(np.where(~overlap_mask)[0], smina_sorted[~overlap_mask], label='SMINA only', color='blue', s=50, alpha=0.6)
plt.scatter(np.where(~overlap_mask)[0], gnina_sorted[~overlap_mask], label='GNINA only', color='orange', s=50, alpha=0.6)

plt.xlabel('Complex Index (Sorted by min(SMINA, GNINA) Rank)', fontsize=18)
plt.ylabel('First Success Rank', fontsize=18)
# plt.title('First Success Pose Rank: SMINA vs GNINA', fontsize=20)
plt.legend(fontsize=14)  
plt.grid(True)
plt.tick_params(labelsize=14)  
plt.show()

# Independent Sorting of First Success Ranks SMINA vs GNINA
# original rank
smina_rank = pose_rank_no_zero['smina_first_success'].values
gnina_rank = pose_rank_no_zero['gnina_first_success'].values

# SMINA ordering
sorted_idx_smina = np.argsort(smina_rank)
smina_sorted_smina = smina_rank[sorted_idx_smina]
gnina_sorted_smina = gnina_rank[sorted_idx_smina]

# GNINA ordering
sorted_idx_gnina = np.argsort(gnina_rank)
smina_sorted_gnina = smina_rank[sorted_idx_gnina]
gnina_sorted_gnina = gnina_rank[sorted_idx_gnina]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].scatter(range(len(smina_sorted_smina)), smina_sorted_smina, label='SMINA', color='blue', s=50)
axes[0].scatter(range(len(gnina_sorted_smina)), gnina_sorted_smina, label='GNINA', color='orange', s=50)
axes[0].set_xlabel('Complex Index (Sorted by SMINA Rank)', fontsize=22)
axes[0].set_ylabel('First Success Rank', fontsize=22)
axes[0].legend(fontsize=20)
axes[0].tick_params(labelsize=20)

axes[1].scatter(range(len(gnina_sorted_gnina)), gnina_sorted_gnina, label='GNINA', color='orange', s=50)
axes[1].scatter(range(len(smina_sorted_gnina)), smina_sorted_gnina, label='SMINA', color='blue', s=50)
axes[1].set_xlabel('Complex Index (Sorted by GNINA Rank)', fontsize=22)
axes[1].legend(fontsize=20)
axes[1].tick_params(labelsize=20)

# plt.suptitle('Independent Sorting of First Success Ranks: SMINA vs GNINA', fontsize=30)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

"""# Compare Smina and Gnina"""

def plot_roc(y_true, scores, label, ax):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    return roc_auc

df_smina = pd.read_csv('./smina_prediction_LEADS_FRAGS_no_overlap.csv')
df_gnina = pd.read_csv('./df_combined_no_overlap.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# smina ROC
auc_smina = plot_roc(df_smina['label'], -df_smina['affinity'], 'SMINA_affinity', ax)

# gnina ROC
auc_gnina_aff = plot_roc(df_gnina['label'], df_gnina['affinity_score'], 'GNINA_affinity_score', ax)
auc_gnina_score = plot_roc(df_gnina['label'], df_gnina['pose_score'], 'GNINA_pose_score', ax)

# diagonal reference
ax.plot([0, 1], [0, 1], 'k--', linewidth=1)

# axis labels and title
ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)
ax.set_title('ROC Curves Comparison: SMINA vs GNINA', fontsize=20)
ax.tick_params(labelsize=14)
ax.legend(loc='lower right', fontsize=14)

plt.tight_layout()
plt.show()

# print AUCs
print(f"AUC smina_affinity: {auc_smina:.3f}")
print(f"AUC gnina_cnnaffinity: {auc_gnina_aff:.3f}")
print(f"AUC gnina_cnnscore: {auc_gnina_score:.3f}")