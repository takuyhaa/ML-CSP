# ML-CSP

## Overview

**ML-CSP** is a machine learning framework designed for crystal structure prediction (CSP) of organic molecules. It provides tools and examples to demonstrate how machine learning techniques can be applied to CSPs, particularly in the context of chemical compound analysis.  
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/0fd53ab8-21d6-4d85-8006-d1bae0079ec6" />


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


## License

This project is licensed under the MIT License. See the LICENSE file for details.
