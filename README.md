![carbonara summary](.img/carbonara_summary.png)

# CARBonAra: Context-aware geometric deep learning for protein sequence design

## Install

The python package and command line tool can be installed using `pip` based on the requirements file: [requirements.txt](requirements.txt). The dependencies are linked with python version 3.9 and can be easily installed using [Anaconda](https://www.anaconda.com/). Create and activate the environment with:

```shell
conda create -n carbonara
conda activate carbonara
pip install .
```

## How to use

### Command line tool

```shell
carbonara --num_sequences 100 --imprint_ratio 0.5 examples/pdbs/2oob.pdb outputs
```

### Python package

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

See [carbonara_quickstart.ipynb](carbonara_quickstart.ipynb) for more details and how to use the package.


## Reproducibility

### Installing Anaconda environment

All the specific dependencies are listed in [carbonara.yml](carbonara.yml). The specific dependencies can be easily installed using [Anaconda](https://www.anaconda.com/). Create and activate the environment with:
```
conda env create -f carbonara.yml
conda activate carbonara
```

### Installing [ESM-IF1](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding) (optional)
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


