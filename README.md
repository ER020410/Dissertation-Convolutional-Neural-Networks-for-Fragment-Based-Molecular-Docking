# Dissertation-Convolutional-Neural-Networks-for-Fragment-Based-Molecular-Docking

This repository contains the code from Yiting Ren's MSc dissertation (P072).  
The project investigates the application of convolutional neural networks (CNNs) for fragment-based molecular docking, focusing on benchmarking, dataset preparation and fine-tuning.

## Setup
!pip install numpy==1.26.4

!pip install molgrid

!pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3

!pip install rdkit

!wget https://drive.usercontent.google.com/u/0/uc?id=1FJGT34IQ23W4dkAKsp8X6i94bmMIJWjV&export=download
!mv uc?id=1FJGT34IQ23W4dkAKsp8X6i94bmMIJWjV smina.static
!chmod 755 smina.static

!apt-get install -y openbabel libopenbabel-dev
!pip install openbabel-wheel
!pip install meeko rdkit-pypi

!pip install -q molgrid gninatorch pytorch-ignite

!pip install mlflow

## Dataset

1. _**LEADS-FRAG**_  
   - Benchmark dataset for fragment docking evaluation.  
   - Paper: LEADS-FRAG: A Benchmark Data Set for Assessment of Fragment Docking Performance,
     Laura Chachulski and Björn Windshügel, Journal of Chemical Information and Modeling 2020 60 (12), 6544-6554, DOI: 10.1021/acs.jcim.0c00693


2. _**Refined PDBBind v2016**_  
   - Public dataset of experimentally determined protein–ligand complexes.  
   - Source: [PDBBind 2016 (via GNINA repository)](https://github.com/gnina/models/tree/master/data/PDBBind2016)  
   - Example file used in this project: `ref_uff_train0.types`

## Scripts

This repository contains four main Python scripts used in the dissertation workflow:

1. **`1_check_test_train_overlap_and_benchmark_for_leads.py`**  
   Checks train/test splits in the LEADS-FRAGS dataset to detect potential overlaps and performs benchmarking of CNN model on filtered _LEADS_FRAG_ dataset.

2. **`2_construction_of_frag_synthetic_dataset.py`**  
   Constructs a synthetic fragment dataset by decomposing ligands into BRICS fragments and assigning labels for CNN training. Produces `.types` files compatible with the GNINA framework.

3. **`3_finetuning_default2018_model.py`**  
   Fine-tunes the pre-trained **default2018** CNN model on the synthetic fragment dataset, supporting different layer-freezing strategies for transfer learning and saving fine-tuned checkpoints.

4. **`4_test_result_on_leads_frags_no_overlap.py`**  
   Evaluates fine-tuned models on the filtered LEADS-FRAGS dataset, computing pose classification metrics and generating results for analysis.

## Models and Utilities

### Initial Model 
- **`crossdock_default2018.pt`**  
  The baseline **default2018** CNN model trained on the _CrossDocked2020_ dataset, used as the model for benchmarking and the starting point for fine-tuning.

### Fine-Tuned Model Variants

The fine-tuned CNN models were trained on the fragment-level dataset with different **layer-freezing strategies** to explore transfer learning performance:

- **Model A (`checkpoint_900_A.pt`)**  
  - Full fine-tuning: all layers of the default2018 network were unfrozen and updated during training.  
  - Tends to adapt strongly to training data but may reduce transferability.

- **Model B (`checkpoint_800_B.pt`)**  
  - Partial fine-tuning: the first **two convolutional units** were frozen, and only deeper layers were updated.  
  - Balances stability with adaptability, achieving intermediate performance.

- **Model C (`checkpoint_900_C.pt`)**  
  - Shallow freezing: the first **four convolutional units** were frozen, fine-tuning only higher-level layers.  
  - Demonstrated the strongest early-ranking ability in pose prediction benchmarks.

These models represent different transfer learning strategies, allowing comparison of generalization and fragment-level docking performance.

### Python Scripts
- **`default2018_model.py`**  
  Defines the architecture of the **default2018** CNN model (GNINA-style 3D convolutional network).

- **`training_finetune.py`**  
  Modified training script for fine-tuning CNN models on fragment datasets, supporting fine-tuning objectives and checkpoint saving.

- **`rmsd_utils.py`**  
  Utility functions for RMSD calculation and pose evaluation, used in benchmarking and model assessment.
