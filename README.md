![carbonara summary](.img/carbonara_summary.png)

# CARBonAra: Context-aware geometric deep learning for protein sequence design

## Overview

CARBonAra is a deep learning framework that facilitates protein sequence design by leveraging atomic coordinates, allowing for context-aware sequence generation. This method is particularly useful for integrating protein design with molecular environments, including non-protein entities, providing more control to protein engineering.

## Features

* **Geometric Transformer**: The framework uses a geometric transformer model based only on atomic coordinates and atomic elements, allowing it to handle any protein backbone scaffolds and various molecular environments.
* **Context Awareness**: CARBonAra's design accounts for molecular environments, including non-protein entities, providing context-aware sequence generation.
* **Imprint Sequence Sampling**: CARBonAra's imprint sampling method provides diverse sequences, balancing design flexibility with high-confidence predictions.

## Install

CARBonAra can be easily installed using `pip` and `conda`:

1. Clone the repository:
```shell
git clone https://github.com/LBM-EPFL/CARBonAra
cd CARBonAra
```

2. Create and activate a new conda environment:
```shell
conda create -n carbonara
conda activate carbonara
```

3. Install the package and dependencies:
```shell
pip install .
```

## Usage

### Command line tool

To generate sequences using a specific protein structure:

```shell
carbonara --num_sequences 100 --imprint_ratio 0.5 examples/pdbs/2oob.pdb outputs
```

### Python package

To use CARBonAra directly in a Python script:

```python
from carbonara import CARBonAra, imprint_sampling

# load model
carbonara = CARBonAra(device_name="cuda")

# sample sequences
sequences, scores, pssm, structure_scaffold = imprint_sampling(
    carbonara=carbonara, 
    pdb_filepath="examples/pdbs/1zns.pdb",  # input structure
    num_sample=100,  # number of sequences to sample
    imprint_ratio=0.5,  # control sampling diversity with prior
)
```
For more detailed examples and use cases, see [quickstart.ipynb](quickstart.ipynb).

### Functionalities

#### required arguments
* `pdb_filepath` (file path): Input scaffold structure file path. The program will automatically remove hydrogens and side chains from the scaffold. The default is to keep the 'hetatm' and water, but this behavior can be toggled.
* `num_sample` (number): Number of sequences to sample using CARBonAra.
* `imprint_ratio` (between 0.0 and 1.0): Ratio of the prediction to use as prior information for sampling. We generate diversity in the sequence space by imprinting predicted prior sequence information at randomly selected positions. The imprinting of predicted prior information creates variability in the final position-specific scoring matrix (PSSM) used to sample sequences. The sequences are sampled from the maximum confidence of the final PSSM to guarantee high-confidence sequences. The imprinting selects how much prior should be added: 0.0 means no prior information, and 1.0 means all the positions will contain prior information to bias the prediction.
#### optional arguments
* `b_sampled` (default: `True`): If true, use the prior sampled from probability for more diversity; if false, use the prior sampled from maximum confidence for sampling. This flag controls the type of prior information imprinted to generate variability in the final position-specific scoring matrix (PSSM). **Sampled from probability** (`True`) will convert the prediction confidence into probabilities and then sample amino acids from those probabilities. It will result in higher diversity while still maintaining reasonable sequence confidence. **Maximum confidence** (`False`) will use the maximum confidence prediction as prior information. It will result in high-confidence predictions but with low diversity.
* `known_chains` (default: `[]`): List of known chains (e.g. `['A', 'B']`) for partial sequence prediction. The software will not predict new sequences for the selected chains. Moreover, the software will use the sequence information of the selected chains as prior information for the prediction.
* `known_positions` (default: `[]`): List of known sequence positions (e.g. `[37, 38, 39, 40]`) for partial sequence prediction. The software will not predict new sequences for the selected sequence positions. Moreover, the software will use the sequence information of the chosen positions as prior information for the prediction.
* `unknown_positions` (default: `[]`): List of unknown sequence positions (e.g. `[37, 38, 39, 40]`) and will overwrite the other known flags. This option should be used when only a relatively small part of the structure should be re-designed. The software will only predict new sequences for the selected sequence positions and will use the rest of the sequence as prior information for the prediction.
* `ignored_amino_acids` (default: `[]`): List of amino acids to completely ignore for the sequence sampling (e.g. `['C']`). The prior information and generated sequences will not contain the selected amino acids.
* `ignore_hetatm` (default: `False`): Flag to ignore hetatm in the structure. By default, the software uses all atoms in the input file for the prediction, such as ligands, lipids, ions, and water.
* `ignore_wat` (default: `False`): Flag to ignore water in the structure. By default, the software uses the water molecules in the input file for the prediction.
* `device` (default: `cuda`): Device to choose for running the model: `cpu` or `cuda`.


## Reproducibility

### Repository structure
* [model](model): Code for the training of the model and the trained model.
* [src](src): Functions used throughout the repository.
* [examples](examples): A few examples of inputs and outputs to get started.
* [results](results): Code to reproduce the results presented in the article.

#### Results
* [model_analysis](results/model_analysis): Analysis and benchmarking of the model, such as the context awareness.
* [model_comparison](results/model_comparison): Comparison with ProteinMPNN and ESM-IF1.
* [md_analysis](results/md_analysis): Analysis of the predictions of CARBonAra when applied to molecular dynamics simulations.
* [cole7](results/cole7): Study case of Colicin E7 (PDB ID: 1ZNS).
* [tem1](results/tem1): Study case of TEM-1 (PDB ID: 1BT5 / 1JTG).

### Anaconda environment

To replicate the specific environment used for development, create and activate it using:

```
conda env create -f carbonara.yml
conda activate carbonara
```

### ESM-IF1 integration

For additional benchmarking with [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding), install it as follow:

```
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg -c conda-forge
conda install pip
pip install ipykernel biotite
pip install git+https://github.com/facebookresearch/esm.git
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Reference


