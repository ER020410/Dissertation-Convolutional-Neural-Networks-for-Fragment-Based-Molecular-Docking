# Dissertation-Convolutional-Neural-Networks-for-Fragment-Based-Molecular-Docking

This repository contains the Jupyter notebooks and results from my MSc dissertation.  
The project investigates the application of convolutional neural networks (CNNs) for fragment-based molecular docking, focusing on dataset preparation, fine-tuning, and benchmarking.

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
1. _LEADS_FRAG_ Dataset: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00693

Citetation: LEADS-FRAG: A Benchmark Data Set for Assessment of Fragment Docking Performance, Laura Chachulski and Björn Windshügel, Journal of Chemical Information and Modeling 2020 60 (12), 6544-6554, DOI: 10.1021/acs.jcim.0c00693

2. Refined _PDBBind v2016_ Dataset: https://github.com/gnina/models/tree/master/data/PDBBind2016 

File Names: ref_uff_train0.types, 

## Dataset

1. _**LEADS-FRAG**_  
   - Benchmark dataset for fragment docking evaluation.  
   - Paper: LEADS-FRAG: A Benchmark Data Set for Assessment of Fragment Docking Performance,
     Laura Chachulski and Björn Windshügel, Journal of Chemical Information and Modeling 2020 60 (12), 6544-6554, DOI: 10.1021/acs.jcim.0c00693


2. _**Refined PDBBind v2016**_  
   - Public dataset of experimentally determined protein–ligand complexes.  
   - Source: [PDBBind 2016 (via GNINA repository)](https://github.com/gnina/models/tree/master/data/PDBBind2016)  
   - Example file used in this project: `ref_uff_train0.types`
