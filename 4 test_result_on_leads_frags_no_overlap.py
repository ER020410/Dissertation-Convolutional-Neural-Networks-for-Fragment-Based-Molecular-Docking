"""Result on LEADS_FRAGS_no_overlap.ipynb

"""

"""# Import Libraries"""
# from google.colab import drive
# drive.mount('/content/drive')

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import os, re
import csv 
import numpy as np
import torch
from scipy.stats import pearsonr
from default2018_model import Net
import molgrid
import pandas as pd

from rmsd_utils import robust_rmsd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

"""# Model A 900 Iterations"""
model_path = "./checkpoint_900.pt"
dims = (28, 48, 48, 48)
model = Net(dims).to('cuda')

def rename_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            new_key = k.replace('features.', '')
        elif k.startswith('pose.'):
            new_key = k.replace('pose.', '')
        else:
            continue  # skip affinity_output 
        new_state_dict[new_key] = v
    return new_state_dict

checkpoint = torch.load(model_path, map_location='cuda')
if 'model' in checkpoint:
    renamed_state_dict = rename_checkpoint_keys(checkpoint['model'])
    model.load_state_dict(renamed_state_dict, strict=False)
else:
    raise ValueError("Checkpoint missing 'model' key")

model.eval()

def get_predictions(model, test_file, label_idx=0, pred_idx=0, batch_size=20, data_root='./'):
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

data_root = "./LEADS_FRAGS_no_overlap"
types_file = os.path.join("./generated_dataset.types")

preds, labels = get_predictions(model, types_file, data_root=data_root, pred_idx=1)

rmse = np.sqrt(np.mean((labels - preds) ** 2))
print(f"default2018 on LEADS_FRAGS → RMSE: {rmse:.3f}")

# load model
dims = (28, 48, 48, 48)
model_path = "/content/checkpoint_900_A.pt"
data_root = "/content/drive/MyDrive/LEADS_FRAGS_no_overlap"
docking_root = "/content/drive/MyDrive/LEADS_FRAGS_no_overlap_docking_results"

model = Net(dims).to('cuda')

def rename_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            new_key = k.replace('features.', '')
        elif k.startswith('pose.'):
            new_key = k.replace('pose.', '')
        else:
            continue  # skip affinity_output
        new_state_dict[new_key] = v
    return new_state_dict

checkpoint = torch.load(model_path, map_location='cuda')
if 'model' in checkpoint:
    model.load_state_dict(rename_checkpoint_keys(checkpoint['model']), strict=False)
else:
    raise RuntimeError("Checkpoint does not contain 'model' key")

model.eval()


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


all_sample_ids = []
all_pose_preds = []
all_labels = []

for idx in complex_ids:
    types_path = os.path.join(data_root, idx, f'{idx}.types')
    if not os.path.exists(types_path):
        print(f"Missing .types file for {idx}, skipping...")
        continue

    pose_preds, labels = get_predictions_pose(model, types_path, label_idx=0, pred_idx=0, batch_size=20, data_root=data_root)

    if len(pose_preds) == 0:
        print(f"No samples loaded for {idx}")
        continue

    all_sample_ids.extend([idx] * len(pose_preds))
    all_pose_preds.extend(pose_preds)
    all_labels.extend(labels)

# Summarize in DataFrame
df = pd.DataFrame({
    'sample_id': all_sample_ids,
    'pose_prediction': all_pose_preds,
    'label': all_labels
})

df.to_csv('./crossdock_0block_900_predictions_LEADS_FRAGS_no_overlap.csv', index=False)
print("Saved predictions to crossdock_0block_900_predictions_LEADS_FRAGS_no_overlap.csv")

crossdock_0block_900_predictions = './crossdock_0block_900_predictions_LEADS_FRAGS_no_overlap.csv'
crossdock_0block_900_predictions = pd.read_csv(crossdock_0block_900_predictions)

# Assess First Success
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

    preds, labels = get_predictions_pose(
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

with open('./crossdock_0block_900_docking_success_results.txt', 'w') as f:
    f.write('\n'.join(results))

print("Saved results to crossdock_0block_900docking_success_results.txt")


input_txt = './crossdock_0block_900_docking_success_results.txt'
output_csv = './crossdock_0block_900_success_pose_ranks.csv'

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

# Pose rank
pose_rank = '/content/drive/MyDrive/crossdock_0block_900_success_pose_ranks.csv'
pose_rank = pd.read_csv(pose_rank)

pose_rank_no_zero = pose_rank[(pose_rank != 0).all(axis=1)]
print(pose_rank_no_zero.head())

# Average pose rank of 1st successful docking for smina and gnina
smina_first_success = pose_rank['smina_first_success'].values
gnina_first_success = pose_rank['gnina_first_success'].values

ave_smina = np.mean(smina_first_success)
ave_gnina = np.mean(gnina_first_success)

print('Average pose rank of 1st successful docking for smina:', ave_smina)
print('Average pose rank of 1st successful docking for gnina:', ave_gnina)

# Top-N rank
def count_success_top_n(pose_rank, rank_col, n=1):
    # only count the success rank equal to n
    success = (pose_rank[rank_col] <= n).sum()
    return success

for n in [1, 3, 5, 10, 15, 20, 25, 30, 34, 35, 36, 37, 38, 39, 40]:
    # print(f'Top {n} success count (smina): {count_success_top_n(pose_rank_no_zero, "smina_first_success", n)}')
    print(f'Top {n} success count (gnina): {count_success_top_n(pose_rank_no_zero, "gnina_first_success", n)}')

"""# Model B 800 Iterations"""
model_path = "/content/checkpoint_800_B.pt"
dims = (28, 48, 48, 48)
model = Net(dims).to('cuda')

def rename_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            new_key = k.replace('features.', '')
        elif k.startswith('pose.'):
            new_key = k.replace('pose.', '')
        else:
            continue  # skip affinity_output
        new_state_dict[new_key] = v
    return new_state_dict

checkpoint = torch.load(model_path, map_location='cuda')
if 'model' in checkpoint:
    renamed_state_dict = rename_checkpoint_keys(checkpoint['model'])
    model.load_state_dict(renamed_state_dict, strict=False)
else:
    raise ValueError("Checkpoint missing 'model' key")

model.eval()

def get_predictions(model, test_file, label_idx=0, pred_idx=0, batch_size=20, data_root='./'):
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

data_root = "./LEADS_FRAGS_no_overlap"
types_file = os.path.join("./generated_dataset.types")

preds, labels = get_predictions(model, types_file, data_root=data_root, pred_idx=1)

rmse = np.sqrt(np.mean((labels - preds) ** 2))
print(f"default2018 on LEADS_FRAGS → RMSE: {rmse:.3f}")

# load model
dims = (28, 48, 48, 48)
model_path = "./checkpoint_800_B.pt"
data_root = "./LEADS_FRAGS_no_overlap"
docking_root = "./LEADS_FRAGS_no_overlap_docking_results"

model = Net(dims).to('cuda')

def rename_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            new_key = k.replace('features.', '')
        elif k.startswith('pose.'):
            new_key = k.replace('pose.', '')
        else:
            continue  # skip affinity_output
        new_state_dict[new_key] = v
    return new_state_dict

checkpoint = torch.load(model_path, map_location='cuda')
if 'model' in checkpoint:
    model.load_state_dict(rename_checkpoint_keys(checkpoint['model']), strict=False)
else:
    raise RuntimeError("Checkpoint does not contain 'model' key")

model.eval()

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


all_sample_ids = []
all_pose_preds = []
all_labels = []

for idx in complex_ids:
    types_path = os.path.join(data_root, idx, f'{idx}.types')
    if not os.path.exists(types_path):
        print(f"Missing .types file for {idx}, skipping...")
        continue

    pose_preds, labels = get_predictions_pose(model, types_path, label_idx=0, pred_idx=0, batch_size=20, data_root=data_root)

    if len(pose_preds) == 0:
        print(f"No samples loaded for {idx}")
        continue

    all_sample_ids.extend([idx] * len(pose_preds))
    all_pose_preds.extend(pose_preds)
    all_labels.extend(labels)

# summarize in DataFrame
df = pd.DataFrame({
    'sample_id': all_sample_ids,
    'pose_prediction': all_pose_preds,
    'label': all_labels
})

df.to_csv('./crossdock_2block_800_predictions_LEADS_FRAGS_no_overlap.csv', index=False)
print("Saved predictions to crossdock_2block_800_predictions_LEADS_FRAGS_no_overlap.csv")

crossdock_2block_800_predictions = './crossdock_2block_800_predictions_LEADS_FRAGS_no_overlap.csv'
crossdock_2block_800_predictions = pd.read_csv(crossdock_2block_800_predictions)

# Assess first success
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

    preds, labels = get_predictions_pose(
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

with open('./crossdock_2block_800_docking_success_results.txt', 'w') as f:
    f.write('\n'.join(results))

print("Saved results to crossdock_2block_800_docking_success_results.txt")

input_txt = './crossdock_2block_800_docking_success_results.txt'
output_csv = './crossdock_2block_800_success_pose_ranks.csv'

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

# pose rank
pose_rank = './crossdock_2block_800_success_pose_ranks.csv'
pose_rank = pd.read_csv(pose_rank)

pose_rank_no_zero = pose_rank[(pose_rank != 0).all(axis=1)]
print(pose_rank_no_zero.head())

# Average pose rank of 1st successful docking for smina and gnina
smina_first_success = pose_rank['smina_first_success'].values
gnina_first_success = pose_rank['gnina_first_success'].values

ave_smina = np.mean(smina_first_success)
ave_gnina = np.mean(gnina_first_success)

print('Average pose rank of 1st successful docking for smina:', ave_smina)
print('Average pose rank of 1st successful docking for gnina:', ave_gnina)

# Top-N rank
def count_success_top_n(pose_rank, rank_col, n=1):
    # only count the success rank equal to n
    success = (pose_rank[rank_col] <= n).sum()
    return success

for n in [1, 3, 5, 10, 15, 20, 25, 30, 34, 35, 36, 37, 38, 39, 40, 45, 49]:
    # print(f'Top {n} success count (smina): {count_success_top_n(pose_rank_no_zero, "smina_first_success", n)}')
    print(f'Top {n} success count (gnina): {count_success_top_n(pose_rank_no_zero, "gnina_first_success", n)}')

"""# Model C 900 Iterations"""

# load model
model_path = "/content/checkpoint_900_C.pt"
dims = (28, 48, 48, 48)
model = Net(dims).to('cuda')

def rename_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            new_key = k.replace('features.', '')
        elif k.startswith('pose.'):
            new_key = k.replace('pose.', '')
        else:
            continue  # skip affinity_output
        new_state_dict[new_key] = v
    return new_state_dict

checkpoint = torch.load(model_path, map_location='cuda')
if 'model' in checkpoint:
    renamed_state_dict = rename_checkpoint_keys(checkpoint['model'])
    model.load_state_dict(renamed_state_dict, strict=False)
else:
    raise ValueError("Checkpoint missing 'model' key")

model.eval()

def get_predictions(model, test_file, label_idx=0, pred_idx=0, batch_size=20, data_root='./'):
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

data_root = "./LEADS_FRAGS_no_overlap"
types_file = os.path.join("./generated_dataset.types")

preds, labels = get_predictions(model, types_file, data_root=data_root, pred_idx=1)

rmse = np.sqrt(np.mean((labels - preds) ** 2))
print(f"default2018 on LEADS_FRAGS → RMSE: {rmse:.3f}")

# load model
dims = (28, 48, 48, 48)
model_path = "./checkpoint_900_C.pt"
data_root = "./LEADS_FRAGS_no_overlap"
docking_root = "./LEADS_FRAGS_no_overlap_docking_results"

model = Net(dims).to('cuda')

def rename_checkpoint_keys(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('features.'):
            new_key = k.replace('features.', '')
        elif k.startswith('pose.'):
            new_key = k.replace('pose.', '')
        else:
            continue  # skip affinity_output
        new_state_dict[new_key] = v
    return new_state_dict

checkpoint = torch.load(model_path, map_location='cuda')
if 'model' in checkpoint:
    model.load_state_dict(rename_checkpoint_keys(checkpoint['model']), strict=False)
else:
    raise RuntimeError("Checkpoint does not contain 'model' key")

model.eval()

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

all_sample_ids = []
all_pose_preds = []
all_labels = []

for idx in complex_ids:
    types_path = os.path.join(data_root, idx, f'{idx}.types')
    if not os.path.exists(types_path):
        print(f"Missing .types file for {idx}, skipping...")
        continue

    pose_preds, labels = get_predictions_pose(model, types_path, label_idx=0, pred_idx=0, batch_size=20, data_root=data_root)

    if len(pose_preds) == 0:
        print(f"No samples loaded for {idx}")
        continue

    all_sample_ids.extend([idx] * len(pose_preds))
    all_pose_preds.extend(pose_preds)
    all_labels.extend(labels)

# summarize in DataFrame
df = pd.DataFrame({
    'sample_id': all_sample_ids,
    'pose_prediction': all_pose_preds,
    'label': all_labels
})

df.to_csv('./crossdock_4block_900_predictions_LEADS_FRAGS_no_overlap.csv', index=False)
print("Saved predictions to crossdock_4block_900_predictions_LEADS_FRAGS_no_overlap.csv")

crossdock_4block_900_predictions = './crossdock_4block_900_predictions_LEADS_FRAGS_no_overlap.csv'
crossdock_4block_900_predictions = pd.read_csv(crossdock_4block_900_predictions)

# Assess first success
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

    preds, labels = get_predictions_pose(
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

with open('./crossdock_4block_900_docking_success_results.txt', 'w') as f:
    f.write('\n'.join(results))

print("Saved results to crossdock_4block_900_docking_success_results.txt")

input_txt = './crossdock_4block_900_docking_success_results.txt'
output_csv = './crossdock_4block_900_success_pose_ranks.csv'

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

# pose rank
pose_rank = './crossdock_4block_900_success_pose_ranks.csv'
pose_rank = pd.read_csv(pose_rank)

pose_rank_no_zero = pose_rank[(pose_rank != 0).all(axis=1)]
print(pose_rank_no_zero.head())

# Average pose rank of 1st successful docking for smina and gnina
smina_first_success = pose_rank['smina_first_success'].values
gnina_first_success = pose_rank['gnina_first_success'].values

ave_smina = np.mean(smina_first_success)
ave_gnina = np.mean(gnina_first_success)

print('Average pose rank of 1st successful docking for smina:', ave_smina)
print('Average pose rank of 1st successful docking for gnina:', ave_gnina)

# Top-N rank
def count_success_top_n(pose_rank, rank_col, n=1):
    # only count the success rank equal to n
    success = (pose_rank[rank_col] <= n).sum()
    return success

for n in [1, 3, 5, 10, 15, 19, 20, 25, 30, 35, 36, 37, 38, 39, 40, 45, 49]:
    # print(f'Top {n} success count (smina): {count_success_top_n(pose_rank_no_zero, "smina_first_success", n)}')
    print(f'Top {n} success count (gnina): {count_success_top_n(pose_rank_no_zero, "gnina_first_success", n)}')

"""# Plot Distribution of Pose Scores"""

gnina_predictions = './gnina_predictions_LEADS_FRAGS_no_overlap.csv'
gnina_predictions = pd.read_csv(gnina_predictions)
pose_scores = gnina_predictions.iloc[:, 1]

palette = sns.color_palette("Set1", 3)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)

# Original GNINA
sns.histplot(pose_scores, bins=40, kde=True, ax=axes[0, 0], color="darkorange")
axes[0, 0].set_title("Distribution of Original GNINA Pose Scores", fontsize=20)
axes[0, 0].set_xlabel("Pose Score", fontsize=18)
axes[0, 0].set_ylabel("Frequency", fontsize=18)
axes[0, 0].tick_params(axis='both', labelsize=16)

# Model A (Full, 900 steps) 
scorea = crossdock_0block_900_predictions.iloc[:, 1]
sns.histplot(scorea, bins=30, kde=True, ax=axes[0, 1], color=palette[0])
axes[0, 1].set_title('Distribution of Model A Pose Scores', fontsize=20)
axes[0, 1].set_xlabel('Pose Score', fontsize=18)
axes[0, 1].set_ylabel('')
axes[0, 1].tick_params(axis='both', labelsize=16)

# Model B (Freeze 1-2, 800 steps)
scoreb = crossdock_2block_800_predictions.iloc[:, 1]
sns.histplot(scoreb, bins=30, kde=True, ax=axes[1, 0], color=palette[1])
axes[1, 0].set_title('Distribtion of Model B Pose Scores', fontsize=20)
axes[1, 0].set_xlabel('Pose Score', fontsize=18)
axes[1, 0].set_ylabel('Frequency', fontsize=18)
axes[1, 0].tick_params(axis='both', labelsize=16)

# Model C (Freeze 1-4, 900 steps) 
scorec = crossdock_4block_900_predictions.iloc[:, 1]
sns.histplot(scorec, bins=30, kde=True, ax=axes[1, 1], color=palette[2])
axes[1, 1].set_title('Distribution of Model C ', fontsize=20)
axes[1, 1].set_xlabel('Pose Score', fontsize=18)
axes[1, 1].set_ylabel('')
axes[1, 1].tick_params(axis='both', labelsize=16)

plt.tight_layout()
plt.show()

"""# Plot Pose Success Rate"""

pose_rank = '/content/drive/MyDrive/success_pose_ranks_no_overlap.csv'
pose_rank = pd.read_csv(pose_rank)
pose_rank_no_zero = pose_rank[(pose_rank != 0).all(axis=1)]

pose_rank_A = './crossdock_0block_900_success_pose_ranks.csv'
pose_rank_A = pd.read_csv(pose_rank_A)
pose_rank_no_zero_A = pose_rank_A[(pose_rank_A != 0).all(axis=1)].reset_index(drop=True)

pose_rank_B = './crossdock_2block_800_success_pose_ranks.csv'
pose_rank_B = pd.read_csv(pose_rank_B)
pose_rank_no_zero_B = pose_rank_B[(pose_rank_B != 0).all(axis=1)].reset_index(drop=True)

pose_rank_C = './crossdock_4block_900_success_pose_ranks.csv'
pose_rank_C = pd.read_csv(pose_rank_C)
pose_rank_no_zero_C = pose_rank_C[(pose_rank_C != 0).all(axis=1)].reset_index(drop=True)

top_n_range = list(range(1, 40))
success_rates_smina = []
success_rates_gnina = []
success_rates_model_A = []
success_rates_model_B = []
success_rates_model_C = []

for n in top_n_range:
    success_smina = count_success_top_n(pose_rank_no_zero, 'smina_first_success', n=n)
    success_gnina = count_success_top_n(pose_rank_no_zero, 'gnina_first_success', n=n)
    success_model_A = count_success_top_n(pose_rank_no_zero_A, 'gnina_first_success', n=n)
    success_model_B = count_success_top_n(pose_rank_no_zero_B, 'gnina_first_success', n=n)
    success_model_C = count_success_top_n(pose_rank_no_zero_C, 'gnina_first_success', n=n)

    total_complexes = 59
    success_rates_smina.append(success_smina / total_complexes)
    success_rates_gnina.append(success_gnina / total_complexes)
    success_rates_model_A.append(success_model_A / total_complexes)
    success_rates_model_B.append(success_model_B / total_complexes)
    success_rates_model_C.append(success_model_C / total_complexes)

plt.figure(figsize=(10, 6))

plt.plot(top_n_range, success_rates_smina, label='SMINA', marker='o')
plt.plot(top_n_range, success_rates_gnina, label='GNINA', marker='s')
plt.plot(top_n_range, success_rates_model_A, label='Model A', marker='^', color = 'purple')
plt.plot(top_n_range, success_rates_model_B, label='Model B', marker='v', color = 'green')
plt.plot(top_n_range, success_rates_model_C, label='Model C', marker='D', color = 'red')

plt.xlabel('Pose Rank Threshold (N)', fontsize=20)
plt.ylabel('Success Rate', fontsize=20)
plt.legend(fontsize=18, loc='lower right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.tick_params(labelsize=16)

plt.tight_layout()
plt.show()

"""# Plot ROC-AUC"""
def plot_roc(y_true, scores, label, ax):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    return roc_auc

df_smina = pd.read_csv('./smina_prediction_LEADS_FRAGS_no_overlap.csv')
df_gnina = pd.read_csv('./df_combined_no_overlap.csv')
df_gnina_0blocks_100 = pd.read_csv('./crossdock_0block_900_predictions_LEADS_FRAGS_no_overlap.csv')
df_gnina_2blocks_100 = pd.read_csv('./crossdock_2block_800_predictions_LEADS_FRAGS_no_overlap.csv')
df_gnina_4blocks_400 = pd.read_csv('./crossdock_4block_900_predictions_LEADS_FRAGS_no_overlap.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# smina ROC
auc_smina = plot_roc(df_smina['label'], -df_smina['affinity'], 'SMINA_affinity', ax)

auc_gnina_score = plot_roc(df_gnina['label'], df_gnina['pose_score'], 'GNINA_score (Original GNINA)', ax)
auc_gnina_score_0blocks_100 = plot_roc(df_gnina_0blocks_100['label'], df_gnina_0blocks_100['pose_prediction'], 'GNINA_score (Model A)', ax)
auc_gnina_score_2blocks_100 = plot_roc(df_gnina_2blocks_100['label'], df_gnina_2blocks_100['pose_prediction'], 'GNINA_score (Model B)', ax)
auc_gnina_score_4blocks_400 = plot_roc(df_gnina_4blocks_400['label'], df_gnina_4blocks_400['pose_prediction'], 'GNINA_score (Model C)', ax)

ax.plot([0, 1], [0, 1], 'k--')

ax.set_xlabel('False Positive Rate', fontsize = 13)
ax.set_ylabel('True Positive Rate', fontsize = 13)
# ax.set_title('ROC Curves Comparison: SMINA vs GNINA (Original and Model A, B, C)', fontsize = 14)
ax.legend(loc='lower right', fontsize = 12)
ax.tick_params(axis='both', labelsize=13)

plt.show()

print(f"AUC SMINA_affinity: {auc_smina:.3f}")
print(f"AUC GNINA_cnnscore (Original): {auc_gnina_score:.3f}")
print(f"AUC GNINA_cnnscore (Model A): {auc_gnina_score_0blocks_100:.3f}")
print(f"AUC GNINA_cnnscore (Model B): {auc_gnina_score_2blocks_100:.3f}")
print(f"AUC GNINA_cnnscore (Model C): {auc_gnina_score_4blocks_400:.3f}")
