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

1. **Start CSP:**

   ```bash
   python ML-CSP/main.py example/benzene/config.yaml
