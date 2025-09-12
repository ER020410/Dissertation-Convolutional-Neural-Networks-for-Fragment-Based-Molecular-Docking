"""Construction of frag_brics dataset.ipynb

# Import
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import Draw
from rdkit.Chem import RDConfig
from rdkit.Chem import BRICS, Descriptors, rdMolDescriptors
from rdkit.Chem import SDMolSupplier, SDWriter
from rdkit.Chem.Draw import rdMolDraw2D

import numpy as np
import pandas as pd
import os, re, shutil
import csv
from collections import defaultdict
from tqdm import tqdm
import random
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
"""

# Prepare Training Set: Receptor and Fragment Ligand
"""
base_dir = './LEADS-FRAGS'
records = []

for pdb_id in os.listdir(base_dir):
    pdb_path = os.path.join(base_dir, pdb_id)

    if os.path.isdir(pdb_path):
        for file_name in os.listdir(pdb_path):
            if file_name.endswith('ligand.mol2'):
                mol2_path = os.path.join(pdb_path, file_name)
                try:
                    mol = Chem.MolFromMol2File(mol2_path, sanitize=True, removeHs=False)
                    if mol is not None:
                        # Calculate the number of atoms and heavy atoms
                        num_atoms = mol.GetNumAtoms()
                        num_heavy_atoms = mol.GetNumHeavyAtoms()

                        # Add the data to the records list
                        records.append({
                            'PDB ID': pdb_id,
                            'File': file_name,
                            'Num Atoms': num_atoms,  # Total number of atoms
                            'Num Heavy Atoms': num_heavy_atoms,  # Number of non-hydrogen (heavy) atoms
                            'Mol Weight': Descriptors.MolWt(mol),
                            'LogP': Descriptors.MolLogP(mol),
                            'TPSA': Descriptors.TPSA(mol),
                            'Num Bonds': mol.GetNumBonds(),
                            'Num Rings': Chem.GetSSSR(mol),
                            'H-bond Acceptors': rdMolDescriptors.CalcNumHBA(mol),
                            'H-bond Donors': rdMolDescriptors.CalcNumHBD(mol),
                            'Aromatic Rings': rdMolDescriptors.CalcNumAromaticRings(mol)
                        })
                except Exception as e:
                    print(f"Error reading {mol2_path}: {e}")

df = pd.DataFrame(records)

df.to_csv('LEADS_FRAGS_properties.csv', index=False)
df = pd.read_csv("./LEADS_FRAGS_properties.csv")

summary = df.describe().T[["min", "max", "mean"]]
"""

# BRICS Fragment Decomposition
"""
ref_type = "./ref_uff_train0.types"

with open(ref_type, "r") as file:
    content = file.read()

# Extract the distinct PDB IDs (before the first slash in the 4th element)
pdb_ids_refine = {line.split()[3].split('/')[0] for line in content.splitlines()}

output_file = './distinct_pdb_ids.txt'
with open(output_file, "w") as f:
    f.write(', '.join(pdb_ids_refine))

print(f"Distinct PDB IDs saved to {output_file}")

# Get refined PDBbind2016
refined_PDBbind2016 = "./refined_PDBbind2016"
output_dir = "./refined_mol2"
os.makedirs(output_dir, exist_ok=True)

pdb_id_folders = [f for f in os.listdir(refined_PDBbind2016) if os.path.isdir(os.path.join(refined_PDBbind2016, f))]

for pdb_id_folder in pdb_id_folders:
    pdb_id_folder_path = os.path.join(refined_PDBbind2016, pdb_id_folder)

    # Look for the _ligand.mol2 file in each pdb_id folder
    for file in os.listdir(pdb_id_folder_path):
        if file.endswith('_ligand.mol2'):
            ligand_file_path = os.path.join(pdb_id_folder_path, file)
            shutil.copy(ligand_file_path, os.path.join(output_dir, file))

print(f"Selected ligand mol2 files have been copied to {output_dir}")

# fragment filtering criteria
criteria = {}
for descriptor in summary.index:
    min_val = summary.loc[descriptor, "min"]
    max_val = summary.loc[descriptor, "max"]
    lower = min_val * 0.9 if min_val != 0 else 0
    upper = max_val * 1.1 if max_val != 0 else 0.1
    criteria[descriptor] = (lower, upper)

# fragment decomposition with filtering
refined_mol2_dir = "./refined_mol2"
all_files = [f for f in os.listdir(refined_mol2_dir) if f.endswith("_ligand.mol2")]

all_files = random.sample(all_files, 1000)

pdb_frag_counter = defaultdict(int)
output_data = []
named_fragments = []
total_mols = 0
matched_mols = 0

def clean_brics_smiles(smi):
    """remove BRICS * and remove Hydrogen"""
    smi = re.sub(r'[0-9]+\*', '*', smi)  # [12*] → [*]
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

for filename in tqdm(all_files):
    mol_path = os.path.join(refined_mol2_dir, filename)
    mol = Chem.MolFromMol2File(mol_path, sanitize=True, removeHs=False)
    if mol is None:
        continue

    total_mols += 1
    pdb_id = filename.split("_")[0]

    fragments = BRICS.BRICSDecompose(
        mol, allNodes=None, minFragmentSize=10, onlyUseReactions=None,
        silent=True, keepNonLeafNodes=False, singlePass=True, returnMols=False
    )

    for frag_smiles in fragments:
        if pdb_frag_counter[pdb_id] >= 2:
            break

        clean_smi = clean_brics_smiles(frag_smiles)
        if clean_smi is None:
            continue

        frag_mol = Chem.MolFromSmiles(clean_smi)
        if frag_mol is None:
            continue

        vec = {
            "Mol Weight": Descriptors.MolWt(frag_mol),
            "TPSA": rdMolDescriptors.CalcTPSA(frag_mol),
            "LogP": Descriptors.MolLogP(frag_mol),
            "Num Atoms": frag_mol.GetNumAtoms(),
            "Num Heavy Atoms": frag_mol.GetNumHeavyAtoms(),
            "Num Bonds": frag_mol.GetNumBonds(),
            "H-bond Acceptors": rdMolDescriptors.CalcNumHBA(frag_mol),
            "H-bond Donors": rdMolDescriptors.CalcNumHBD(frag_mol),
            "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(frag_mol)
        }

        if all(criteria[key][0] <= val <= criteria[key][1] for key, val in vec.items()):
            frag_index = pdb_frag_counter[pdb_id] + 1
            frag_name = f"{pdb_id}_ligand_frag_{frag_index}"

            output_data.append([filename, clean_smi, *vec.values(), frag_name])
            named_fragments.append((clean_smi, frag_name))
            pdb_frag_counter[pdb_id] += 1

    matched_mols += 1

print(f"Total ligands processed: {total_mols}")
print(f"Fragments matched criteria: {len(output_data)}")

df = pd.DataFrame(output_data, columns=[
    'ligand_file', 'fragment_smiles',
        'Mol Weight', 'TPSA', 'LogP', 'Num Atoms', 'Num Heavy Atoms',
        'Num Bonds', 'H-bond Acceptors', 'H-bond Donors', 'Aromatic Rings', 'fragment_name'
])
csv_file = "./filtered_fragments_refined.csv"
df.to_csv(csv_file, index=False)
print(f"Saved CSV to {csv_file}")

frag_brics = pd.read_csv("./filtered_fragments_refined.csv")
"""

# Example graph
"""
# read orginal ligand .mol2 file
ligand = Chem.MolFromMol2File("/content/drive/MyDrive/1a0q/1a0q_ligand.mol2", sanitize=True, removeHs=False)

# BRICS fragment decomposition
brics_frags = BRICS.BRICSDecompose(ligand, minFragmentSize=4, onlyUseReactions=None, silent=True, keepNonLeafNodes=True, singlePass=True, returnMols=False)
brics_frags = list(brics_frags)[:3]  

# SMILES to Mol 
frag_mols = [Chem.MolFromSmiles(frag) for frag in brics_frags]

all_mols = [ligand] + frag_mols
titles = ["Original Ligand", "Fragment 1", "Fragment 2", "Fragment 3"]

img = Draw.MolsToImage(all_mols, molsPerRow=len(all_mols), subImgSize=(300,300), legends=titles)
img.save("brics_fragment_example.png")
img.show()
"""

# To SMILE file"""
frag_brics = pd.read_csv("./filtered_fragments_refined.csv")
frag_brics = frag_brics.sort_values(by=['ligand_file'])
frag_brics

csv_path = "./filtered_fragments_refined.csv"
out_smi  = "./filtered_fragments_refined.smi"

df = pd.read_csv(csv_path)

def clean_brics_smiles(smi):
    smi = re.sub(r"\[\d+\*\]", "[*]", smi)
    smi = re.sub(r"\(\)", "", smi) 
    return smi

ligand_counter = defaultdict(int)
rows = []

for i, r in df.iterrows():
    smi = str(r["fragment_smiles"]).strip()
    base_name = str(r["fragment_name"])
    clean_smi = clean_brics_smiles(smi)

    ligand_counter[base_name] += 1
    numbered_name = f"{base_name}"

    rows.append((clean_smi, numbered_name))

with open(out_smi, "w") as f:
    for smi, name in rows:
        f.write(f"{smi}\t{name}\n")

print(f"Output saved to: {out_smi}")
print(f"Total fragments: {len(df)}")

"""## Alignment with _docked.sdf"""

# !mkdir -p /content/PDBbind2016
# !tar -xzf /content/drive/MyDrive/PDBbind2016.tar.gz -C /content/PDBbind2016

# matched fragment content
refined_dir = "/content/drive/MyDrive/aligned_fragments_refined2"

# original PDBbind2016 refined set
pdbbind2016_dir = "/content/PDBbind2016"

# get matched PDB IDs
matched_pdb_ids = [d for d in os.listdir(refined_dir) if os.path.isdir(os.path.join(refined_dir, d))]

print(f"Found {len(matched_pdb_ids)} matched PDB IDs.")

copied = 0
missing = []

for pdb_id in tqdm(matched_pdb_ids, desc="Copying docked.sdf to matched folders"):
    src_path = os.path.join(pdbbind2016_dir, pdb_id, f"{pdb_id}_docked.sdf")
    dst_path = os.path.join(refined_dir, pdb_id, f"{pdb_id}_docked.sdf")  # new name

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied += 1
    else:
        missing.append(pdb_id)

print(f"\n Copied {copied} docked.sdf files into matched fragment folders")
if missing:
    print(f"Missing {len(missing)} docked.sdf files: {missing[:10]} ...")


def align_frag_to_all_poses(poses, smi_frag, sdwriter, frag_id=None, output_dir=None):
    matched_pose_ids = []
    """
    Aligns the 3D coordinates of a fragment (from SMILES) to every conformer (pose)
    of a docked molecule (from _docked.sdf).

    Arguments:
        - mol_poses: list of mol objects (each with 3D coordinates)
        - smi_frag: fragment SMILES with exit vector (e.g. [*])
        - sdwriter: open SDWriter to write aligned fragments
    """
    for idx, mol in enumerate(poses):
        try:
            mol = Chem.RemoveHs(mol)
            frag = Chem.RemoveHs(Chem.MolFromSmiles(smi_frag))

            # Adjust query for dummy atom matching
            qp = Chem.AdjustQueryParameters()
            qp.makeDummiesQueries = True
            qfrag = Chem.AdjustQueryProperties(frag, qp)

            # Try substructure match (frag in mol)
            frag_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
            if len(frag_matches) == 0:
                print(f"No match in pose {idx}")
                continue

            matched_pose_ids.append(f"{frag_id}_pose{idx}")

            frag_match = frag_matches[0]  # use the first match
            sub_idx = frag_match + tuple(i for i in range(mol.GetNumAtoms()) if i not in frag_match)

            # Renumber atoms in mol and frag for alignment
            mol_reorder = Chem.rdmolops.RenumberAtoms(mol, sub_idx)
            aligned_mols = [mol_reorder, frag]

            # Move dummy atoms to end (frag only)
            dummy_idx = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
            sub_idx_frag = list(range(frag.GetNumAtoms()))
            for idx in dummy_idx:
                sub_idx_frag.remove(idx)
            sub_idx_frag.extend(dummy_idx)

            aligned_mols[1] = Chem.rdmolops.RenumberAtoms(frag, sub_idx_frag)

            # Assign reference coordinates to fragment
            ref_conf = aligned_mols[0].GetConformer()
            conf = Chem.Conformer(frag.GetNumAtoms())
            for i in range(frag.GetNumAtoms()):
                pos = list(ref_conf.GetAtomPosition(i))
                conf.SetAtomPosition(i, pos)
            aligned_mols[1].AddConformer(conf)

            # Align fragment to mol using heavy atoms
            AllChem.AlignMol(aligned_mols[1], aligned_mols[0], atomMap=[(i, i) for i in range(frag.GetNumAtoms())])

            # Remove dummy atoms
            aligned_clean = Chem.RemoveHs(AllChem.ReplaceSubstructs(
                aligned_mols[1],
                Chem.MolFromSmiles('*'),
                Chem.MolFromSmiles('[H]'),
                True
            )[0])

            # Annotate name and write out
            aligned_clean.SetProp("_Name", f"{smi_frag}_pose{idx}")
            sdwriter.write(aligned_clean)

        except Exception as e:
            print(f"Failed on pose {idx}: {smi_frag} — {str(e)}")
            continue
    # Save matched poses list
    if output_dir and frag_id and matched_pose_ids:
        matched_txt_path = os.path.join(output_dir, f"{frag_id}_matched_poses.txt")
        with open(matched_txt_path, "w") as f:
            for pose_name in matched_pose_ids:
                f.write(pose_name + "\n")
        print(f"Saved matched poses to {matched_txt_path}")

def batch_align(smi_file, root_dir, output_dir):
    success = 0
    fail = 0
    success_pdb_ids = set()
    fail_pdb_ids = set()
    matched_pose_ids = []

    with open(smi_file) as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="Aligning fragments"):
        try:
            smi, frag_id = line.strip().split()
            pdb_id = frag_id.split("_")[0]
        except:
            print(f"Invalid line format: {line}")
            fail += 1
            continue

        sdf_path = os.path.join(root_dir, pdb_id, f"{pdb_id}_docked.sdf")
        out_path = os.path.join(root_dir, pdb_id, f"{frag_id}_aligned_new.sdf")

        if not os.path.exists(sdf_path):
            print(f"Missing file: {sdf_path}")
            fail += 1
            fail_pdb_ids.add(pdb_id)
            continue

        poses = SDMolSupplier(sdf_path, removeHs=False)
        poses = [m for m in poses if m is not None]

        if len(poses) == 0:
            print(f"No valid poses in {sdf_path}")
            fail += 1
            fail_pdb_ids.add(pdb_id)
            continue

        try:
            writer = SDWriter(out_path)
            align_frag_to_all_poses(poses, smi, writer, frag_id=frag_id,
                output_dir=os.path.join(root_dir, pdb_id))
            writer.close()
            success += 1
            success_pdb_ids.add(pdb_id)
        except Exception as e:
            print(f"Failed to align {frag_id}: {e}")
            fail += 1
            fail_pdb_ids.add(pdb_id)

    # Save logs
    success_list_file = output_dir.replace('.sdf', '_success_pdb_ids.txt')
    fail_list_file = output_dir.replace('.sdf', '_fail_pdb_ids.txt')

    with open(success_list_file, "w") as f:
        for pdb in sorted(success_pdb_ids):
            f.write(pdb + "\n")

    with open(fail_list_file, "w") as f:
        for pdb in sorted(fail_pdb_ids):
            f.write(pdb + "\n")

    # Print summary
    print(f"\n Alignment finished: {success} succeeded, {len(success_pdb_ids)} unique PDB IDs")
    print(f" Alignment failed:   {fail} failed, {len(fail_pdb_ids)} unique PDB IDs")
    print(f" Output saved to: {output_dir}")
    print(f" Success PDB list: {success_list_file}")
    print(f" Failed PDB list:  {fail_list_file}")

batch_align(
    smi_file = "./filtered_fragments_refined.smi",
    root_dir="./aligned_fragments_refined2",
    output_dir="./all_aligned_fragments.sdf"
)

root_dir = "./aligned_fragments_refined2"

for pdb_id in tqdm(os.listdir(root_dir), desc="Processing PDB folders"):
    pdb_dir = os.path.join(root_dir, pdb_id)
    if not os.path.isdir(pdb_dir):
        continue

    for fname in os.listdir(pdb_dir):
        if not fname.endswith("_aligned.sdf"):
            continue

        sdf_path = os.path.join(pdb_dir, fname)

        try:
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        except:
            print(f"Error reading {sdf_path}")
            continue

        mols = [mol for mol in suppl if mol is not None]
        if not mols:
            continue

        base_name = fname.replace("_aligned.sdf", "")
        for idx, mol in enumerate(mols):
            out_name = f"{base_name}_pose{idx}.sdf"
            out_path = os.path.join(pdb_dir, out_name)
            writer = Chem.SDWriter(out_path)
            writer.write(mol)
            writer.close()

print("All poses have been split into individual SDF files")

"""# Prepare Training Set: Receptor and Fragment Ligand

## Check train/test overlap
"""
root_dir = "./aligned_fragments_refined2"
pdb_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
pdb_ids_upper = set(id_.upper() for id_ in pdb_ids)
pdb_ids = pd.DataFrame(sorted(pdb_ids_upper), columns=['sample_id'])

leads_frag_root = './LEADS-FRAGS'
leads_sample_ids = set(os.listdir(leads_frag_root))
leads_sample_ids_upper = set(id_.upper() for id_ in leads_sample_ids)
df_leads = pd.DataFrame(sorted(leads_sample_ids_upper), columns=['sample_id'])
print(df_leads.head())

overlap = pdb_ids_upper.intersection(leads_sample_ids_upper)

print(f'Number of overlapping sample_ids: {len(overlap)}')

"""## Prepare Corresponding Receptor"""

# Existing matched fragment list
matched_dir = "./aligned_fragments_refined2"

# Original PDBbind2016 refined set
pdbbind2016_dir = "./PDBbind2016"

# Get matched PDB IDs
matched_pdb_ids = [d for d in os.listdir(matched_dir) if os.path.isdir(os.path.join(matched_dir, d))]

print(f"Found {len(matched_pdb_ids)} matched PDB IDs.")

copied = 0
missing = []

for pdb_id in tqdm(matched_pdb_ids, desc="Copying _rec_0.gninatypes to matched folders"):
    src_path = os.path.join(pdbbind2016_dir, pdb_id, f"{pdb_id}_rec_0.gninatypes")
    dst_path = os.path.join(matched_dir, pdb_id, f"{pdb_id}_rec_0.gninatypes")  # new name

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied += 1
    else:
        missing.append(pdb_id)

print(f"\n Copied {copied} _rec_0.gninatypes files into matched fragment folders")
if missing:
    print(f"Missing {len(missing)} _rec_0.gninatypes files: {missing[:10]} ...")

"""# Label Construction"""

ref_type = "./ref_uff_train0.types"

with open(ref_type, "r") as file:
    lines = [file.readline() for _ in range(20)]
for line in lines:
    print(line.strip())

def generate_labels_for_split_poses(matched_root, type_file):
    # read gninatypes label 
    pose_to_label = {}
    with open(type_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            label = parts[0]
            pose_path = parts[4]  # e.g. 3ta1/3ta1_docked_0.gninatypes
            pose_to_label[pose_path] = label

    print(f"Loaded {len(pose_to_label)} labels from types file")

    # for evert PDB folder
    for pdb_id in tqdm(os.listdir(matched_root), desc="Processing folders"):
        frag_dir = os.path.join(matched_root, pdb_id)
        if not os.path.isdir(frag_dir):
            continue

        # poseX.sdf 
        all_pose_files = [f for f in os.listdir(frag_dir) if f.endswith(".sdf") and "_pose" in f]

        frag_to_poses = {}
        for f in all_pose_files:
            base = "_".join(f.split("_")[:4])  # e.g. 3ta1_ligand_frag_1
            frag_to_poses.setdefault(base, []).append(f)

        # for every fragment assign label file
        for base_name, pose_files in frag_to_poses.items():
            label_file = os.path.join(frag_dir, f"{base_name}_labels.txt")
            with open(label_file, "w") as lf:
                for pose_file in sorted(pose_files):
                    # get pose index
                    try:
                        idx = pose_file.split("_pose")[-1].replace(".sdf", "")
                        pose_key = f"{pdb_id}/{pdb_id}_docked_{idx}.gninatypes"
                        label = pose_to_label.get(pose_key, "NA")
                        lf.write(f"{pose_file} {label}\n")
                    except Exception as e:
                        lf.write(f"{pose_file} NA\n")

generate_labels_for_split_poses(
    matched_root="./aligned_fragments_refined2",
    type_file="./ref_uff_train0.types")

"""# Training Setup

## Get RMSD score
"""
types_file = "./ref_uff_train0.types"
output_csv = "./rmsd_values.csv"

records = []

with open(types_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        rmsd = float(parts[2])
        ligand_path = parts[4]

        # Get pdb_id, fragment_id and pose_idx
        pdb_id = ligand_path.split('/')[0]
        filename = os.path.basename(ligand_path)

        if "_docked_" not in filename:
            continue 

        fragment_id = filename.split('_docked_')[0] + "_docked"
        pose_idx = int(filename.split('_docked_')[1].split('.')[0])

        records.append({
            "pdb_id": pdb_id,
            "fragment_id": fragment_id,
            "pose_idx": pose_idx,
            "rmsd": rmsd
        })

# CSV file
with open(output_csv, "w", newline="") as csvfile:
    fieldnames = ["pdb_id", "fragment_id", "pose_idx", "rmsd"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print(f"Extracted {len(records)} entries to {output_csv}")

rsmd_df = pd.read_csv("./rmsd_values.csv")

# Construct a DataFrame for frag_1 and frag_2
frag_1 = rsmd_df.copy()
frag_1["fragment_id"] = frag_1["pdb_id"] + "_ligand_frag_1"

frag_2 = rsmd_df.copy()
frag_2["fragment_id"] = frag_2["pdb_id"] + "_ligand_frag_2"

# combine
final_df = pd.concat([frag_1, frag_2], ignore_index=True)
final_df = final_df[["pdb_id", "fragment_id", "pose_idx", "rmsd"]]

output_path = "./rmsd_values.csv"
final_df.to_csv(output_path, index=False)

pd.read_csv(output_path)

matched_root = "./aligned_fragments_refined2"

# read rmsd csv
rmsd_csv = "./rmsd_values.csv"
rmsd_df = pd.read_csv(rmsd_csv)

# get matched_root's pdb_id
pdb_ids = [name for name in os.listdir(matched_root)
           if os.path.isdir(os.path.join(matched_root, name))]

# filter pdb_id in RMSD file 
filtered_rmsd = rmsd_df[rmsd_df["pdb_id"].isin(pdb_ids)]

# save new CSV
output_path = "./rmsd_values.csv"
filtered_rmsd.to_csv(output_path, index=False)

print(f"Filtered {len(filtered_rmsd)} entries and saved to: {output_path}")
pd.read_csv(output_path)

"""## Get Vina Score"""
def generate_vina_scores_per_pdb(matched_root, types_file):
    # pose_key to vina_score
    pose_to_vina = {}
    with open(types_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            pose_key = parts[4]      # e.g. 1a4k/1a4k_docked_0.gninatypes
            vina_score = parts[-1]   # vina score is the last
            pose_to_vina[pose_key] = vina_score

    print(f"Loaded {len(pose_to_vina)} vina scores")

    # for every PDB folder
    for pdb_id in os.listdir(matched_root):
        pdb_dir = os.path.join(matched_root, pdb_id)
        if not os.path.isdir(pdb_dir):
            continue

        # get all label.txt
        label_files = [f for f in os.listdir(pdb_dir) if f.endswith("_labels.txt")]
        for label_file in label_files:
            label_path = os.path.join(pdb_dir, label_file)

            # output path
            base_name = label_file.replace("_labels.txt", "")
            out_file = os.path.join(pdb_dir, f"{base_name}_vina_scores.txt")

            with open(label_path) as lf, open(out_file, "w") as out:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    pose_sdf = parts[0]  # e.g. 1a4k_ligand_frag_1_pose0.sdf

                    # get pose_key
                    pdb_id = pose_sdf.split("_")[0]
                    pose_idx = pose_sdf.split("_pose")[-1].replace(".sdf", "")
                    pose_key = f"{pdb_id}/{pdb_id}_docked_{pose_idx}.gninatypes"

                    vina_score = pose_to_vina.get(pose_key, "NA")
                    out.write(f"{pose_sdf} {vina_score}\n")

            print(f"Wrote vina scores to {out_file}")

matched_root = "./aligned_fragments_refined2"
types_file = "./ref_uff_train0.types"

generate_vina_scores_per_pdb(matched_root, types_file)

"""## .type file"""

def generate_final_types(matched_root, rmsd_csv, output_types):
    # read RMSD list (pdb_id, fragment_id, pose_idx to rmsd)
    rmsd_map = {}
    with open(rmsd_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['pdb_id']}/{row['fragment_id']}_pose{row['pose_idx']}.sdf"
            rmsd_map[key] = row['rmsd']

    rows = []

    # for every pdb_id folder
    for pdb_id in os.listdir(matched_root):
        pdb_dir = os.path.join(matched_root, pdb_id)
        if not os.path.isdir(pdb_dir):
            continue

        # find receptor
        rec_files = [f for f in os.listdir(pdb_dir) if f.endswith("_rec_0.gninatypes")]
        if not rec_files:
            continue
        receptor = f"{pdb_id}/{rec_files[0]}"

        # for label.txt
        for f in os.listdir(pdb_dir):
            if not f.endswith("_labels.txt"):
                continue

            frag_base = f.replace("_labels.txt", "")
            label_file = os.path.join(pdb_dir, f)
            vina_file  = os.path.join(pdb_dir, f"{frag_base}_vina_scores.txt")

            # read vina score
            pose_to_vina = {}
            if os.path.exists(vina_file):
                with open(vina_file) as vf:
                    for line in vf:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            pose_to_vina[parts[0]] = parts[1]

            # read label + RMSD + vina
            with open(label_file) as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    pose_sdf, label = parts
                    ligand = f"{pdb_id}/{pose_sdf}"
                    rmsd   = rmsd_map.get(f"{pdb_id}/{pose_sdf}", "NA")
                    vina   = pose_to_vina.get(pose_sdf, "NA")

                    # affinity fix to 0
                    rows.append(f"{label} 0 {rmsd} {receptor} {ligand} # {vina}")

    # final types file
    with open(output_types, "w") as f:
        f.write("\n".join(rows))

    print(f"Wrote {len(rows)} lines to {output_types}")

matched_root = "./aligned_fragments_refined2"
rmsd_csv = "./rmsd_values.csv"
output_types = "./final_dataset.types"

generate_final_types(matched_root, rmsd_csv, output_types)

train_type = "./final_dataset.types"

with open(train_type, "r") as file:
    lines = [file.readline() for _ in range(100)]
for line in lines:
    print(line.strip())

# .type file statistics
types_file = "./final_dataset.types"

# initialize
pdb_to_frags = defaultdict(set)
frag_to_pose_count = defaultdict(int)
total_poses = 0

with open(types_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        ligand_path = parts[4]

        # example: 2qmg/2qmg_ligand_frag_1_pose0.sdf
        match = re.match(r"([^/]+)/\1_ligand_(frag_\d+)_pose\d+\.sdf", ligand_path)
        if match:
            pdb_id = match.group(1)
            frag_id = match.group(2)
            pdb_to_frags[pdb_id].add(frag_id)
            frag_to_pose_count[(pdb_id, frag_id)] += 1
            total_poses += 1

# statsitics
pdb_with_2_frags = sum(1 for frags in pdb_to_frags.values() if len(frags) == 2)
pdb_with_1_frag = sum(1 for frags in pdb_to_frags.values() if len(frags) == 1)
total_pdb = len(pdb_to_frags)
total_frags = len(frag_to_pose_count)

# print the result
print(f"Total unique PDB IDs: {total_pdb}")
print(f"PDBs with 2 fragments: {pdb_with_2_frags}")
print(f"PDBs with 1 fragment: {pdb_with_1_frag}")
print(f"Total unique fragments: {total_frags}")
print(f"Total poses: {total_poses}")

# count the number of pose for every fragment
df_pose_counts = pd.DataFrame([
    {"PDB_ID": pdb, "Fragment": frag, "Pose_Count": count}
    for (pdb, frag), count in frag_to_pose_count.items()
])

df_pose_counts.head()

df_pose_counts["Pose_Count"].describe()

def count_unique_pdb_ids(file_path):
    with open(file_path, 'r') as f:
        pdb_ids = {line.strip() for line in f if line.strip()}
    return pdb_ids

success_file = './all_aligned_fragments_success_pdb_ids.txt'
failed_file = './all_aligned_fragments_fail_pdb_ids.txt'

success_pdbs = count_unique_pdb_ids(success_file)
failed_pdbs = count_unique_pdb_ids(failed_file)

print(f"Unique PDB IDs in success.txt: {len(success_pdbs)}")
print(f"Unique PDB IDs in failed.txt: {len(failed_pdbs)}")
print(f"Total unique PDB IDs: {len(success_pdbs.union(failed_pdbs))}")
