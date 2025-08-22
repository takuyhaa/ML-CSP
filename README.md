# ML-CSP

## Overview

**ML-CSP** is a machine learning framework designed for crystal structure prediction (CSP) of organic molecules. It provides tools and examples to demonstrate how machine learning techniques can be applied to CSPs, particularly in the context of chemical compound analysis.  
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/0fd53ab8-21d6-4d85-8006-d1bae0079ec6" />

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/takuyhaa/ML-CSP.git
   cd ML-CSP

1. **Set Up the Conda Environment**
   ```bash
   conda env create -f environment.yml
   conda activate mlcsp

1. **Download Pre-trained Models**  
   ML models are provided via GitHub Large File System (LFS):
   ```bash
   # Install git-lfs (for Ubuntu/Debian)
   sudo apt-get update
   sudo apt-get install git-lfs
   
   # Initialize and pull LFS files
   git lfs install
   git lfs pull
   ```

1. **Install Additional Dependencies (Optional)**  
   Some packages may require manual installation:
   ```bash
   # If you encounter ModuleNotFoundError during execution
   pip install chgnet  # For CHGNet potential
   pip install torchani  # For ANI potential
   ```
   *PFP is available on Matlantis and requires a valid license to use. For more information on obtaining a license and using PFP, please visit the Matlantis website.

## Quick Start Guide
### Running the Benzene Example
```bash
cd ML-CSP
python main.py ../example/benzene/config.yaml
```
This will:
1. Generate conformers from SMILES (c1ccccc1 for benzene)
1. Create 10 crystal structures with random space groups and Z' values
1. Optimize structures using CHGNet
1. Save results to example/benzene/ directory

### Expected Output  
After successful execution, you'll find in example/benzene/:
- xyz/: Generated molecular conformers
- init_structures.cif: Initial crystal structures before optimization
- init_structures.pkl: Initial crystal structures before optimization
- opt_structures.cif: Optimized crystal structures
- opt_structures.pkl: Optimized crystal structures
- opt_results.csv: Energy, density, and space group information

The console will show:
```bash
Molecular Weight: 78.114 g/mol
Predicted density: X.XXX g/cm3
SG candidates: [1, 2, 4, 5, 7, ...]
Z candidates: [1, 2, 3, 4, 6, 8, 9, 16]
5 conformers found (n_confs:5)
ID_0 (SG: 14, numIons: 4, n_atoms: 12, n_asym_unit: 12) generated
...
10 have been optimized. (Total: XXX.X sec)
```

## Configuration File Structure  
Create a YAML configuration file for your molecule:
```yaml
# Molecular Input
smiles: "your_smiles_string"           # SMILES notation of your molecule

# Conformer Generation
conformer_mode: search                 # 'search' or 'predefined'
num_conformers: 5                      # Number of conformers to generate

# Structure Generation
num_structures: 100                    # Number of crystal structures to generate
list_numIons: [1, 2, 3, 4, 6, 8]      # Possible Z' values (molecules per asymmetric unit)

# ML-Guided Generation Modes
ml_mode: proba_threshold_random        # ML probability mode
sg_mode: ml                           # 'ml', 'random', 'sg95', or specific SG number
Z_mode: ml                            # 'ml', 'random', or 'sg_dependent'
density_mode: ml                      # 'ml' or 'random'
lattice_mode: VAE                     # 'VAE' or 'None' for random

# Optimization
model_name: CHGNet                    # 'CHGNet', 'ANI', or 'PFP'

# Output
root_folder: results/your_molecule/   # Output directory path
```

## How to Use Your Own Crystal Structures  
### Option 1: Starting from SMILES  
1. Create a configuration file (your_molecule_config.yaml):
   ```yaml
   smiles: "CC(C)CC(C)C(=O)O"  # Your molecule's SMILES
   conformer_mode: search
   num_conformers: 10
   num_structures: 200
   list_numIons: [1, 2, 4]
   model_name: CHGNet
   ml_mode: proba_threshold_random
   sg_mode: ml
   Z_mode: ml
   density_mode: ml
   lattice_mode: None
   root_folder: results/my_molecule/
   ```
1. Run the prediction:
   ```bash
   python main.py your_molecule_config.yaml
   ```

### Option 2: Using Pre-existing Conformers
1. Place your conformer XYZ files in the xyz/ subdirectory
1. Set conformer_mode: predefined in config
1. Run as usual

### Option 3: Working with CSD Structures
1. Download CIF from CSD
1. Extract molecular geometry and save as XYZ
1. Place in xyz/ directory
1. Configure and run






<!-- 
## Repository Structure

- `datasets/`: Contains datasets used for training and evaluating machine learning models.
- `example/benzene/`: Provides an example application of ML-CSP on benzene molecules.
- `notebook/`: Includes Jupyter notebooks for exploratory data analysis and model development.
- `environment.yml`: Specifies the conda environment configuration for reproducing the development environment.
- `README.md`: This file, providing an overview and instructions for the ML-CSP project.

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/takuyhaa/ML-CSP.git
   cd ML-CSP

1. **Set up the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate mlcsp

1. **Download models:**
   
   ```bash
   sudo apt-get install git-lfs
   git lfs install
   git lfs pull
   ```
   ML models are provided via GitHub Large File System (LFS). So, you need to set up git-lfs.  
   The above command is available for Ubuntu/Debian.


1. **Start CSP:**

   ```bash
   cd ML-CSP
   python main.py ../example/benzene/config.yaml
   ```
   The excecution may arise import error.  
   If you encounter `ModuleNotFoundError`, please excecute `pip install MODULE`.
   

## Neural Network Potential
ML-CSP incorporates three Neural Network Potentials (NNPs) for structure relaxation:

1. **CHGNet**: A pretrained universal neural network potential for charge-informed atomistic modeling.
2. **ANI**: An extensible neural network potential with DFT accuracy at force field computational cost.
3. **PFP (PreFerred Potential)**: A universal neural network potential developed by Matlantis, requiring a valid license for use.

### CHGNet
To install CHGNet, you can use pip:

```bash
pip install chgnet
```

### ANI

To install ANI, you can use pip:

```bash
pip install torchani
```

### PFP
PFP is available on Matlantis and requires a valid license to use. For more information on obtaining a license and using PFP, please visit the Matlantis website.
-->

## License

This project is licensed under the MIT License. See the LICENSE file for details.
