# PUMA: A Perturbation-Augmented Unsupervised Learning Framework for Integer Linear Programming

This repository contains the official implementation of our KDD 2026 paper,
**"A Perturbation-Augmented Unsupervised Learning Framework for Integer Linear Programming."**

The paper is published at the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data
Mining (KDD 2026), DOI [10.1145/3770855.3817737](https://doi.org/10.1145/3770855.3817737).

## Project Structure

```
.
├── config/                 # Hydra config files
│   ├── dataset/            #   per-dataset settings (CA, IS, SC)
│   ├── model/              #   GNN architecture (GCN/GAT, depth, embd_size)
│   ├── trainer/            #   training hyperparameters (μ schedule, p_flip, num_samples)
│   ├── paths/              #   input/output path settings
│   ├── hydra/              #   Hydra runtime config
│   ├── train.yaml          #   training entry config
│   ├── test.yaml           #   evaluation entry config
│   ├── preprocess.yaml     #   preprocessing entry config
│   └── evaluation.yaml     #   evaluation/reporting config
├── scripts/                # Environment setup + example train/test commands
├── src/
│   ├── model.py            #   Bipartite GNN (GCN/GAT) predictor, dataset, data classes
│   ├── solver_wrapper.py   #   Unified Gurobi / SCIP interface (warm start, local branching)
│   ├── utils.py            #   Graph extraction, Gumbel-Softmax sampling, solving utilities
│   └── tb_writter.py       #   TensorBoard logging helper
├── preprocess.py           # Data preprocessing (instances -> bipartite tensors)
├── train.py                # Training entry point
├── test.py                 # Evaluation entry point (warm start + local branching)
└── README.md
```

## Environment Setup

- Python environment
    - python 3.8
    - pytorch 2.3.0
    - torch-geometric 2.6
    - ecole 0.8.1
    - pyscipopt 4.4.0
    - gurobipy 10.0+
    - tensorboardX, hydra-core, omegaconf, tqdm

- MILP Solver
    - [Gurobi](https://www.gurobi.com/) 11.0.1 (Academic License). SCIP 5.4.0 is also supported.

- [Hydra](https://hydra.cc/docs/intro/) is used for managing hyperparameters and experiments.

To set up the environment, run the installation script:
```bash
bash scripts/environment.sh
```
Alternatively, create the environment from a file:
```bash
conda env create -f scripts/environment.yml
```

## Data Format

Place datasets under `data/<NAME>/train/` and `data/<NAME>/test/`. Instances may be in any
format readable by SCIP/ecole (e.g. `.lp` or `.mps`). Configurations are provided for the
three problem classes used in the paper: CA, IS, and SC.

## Usage

All commands are driven by Hydra; override any field on the command line.

### 1. Preprocessing
Convert raw instances into bipartite-graph tensors (e.g. for `SC`):
```bash
python preprocess.py dataset=SC num_workers=50
```

### 2. Training
```bash
python train.py dataset=SC cuda=0
```
Key hyperparameters live in `config/trainer/<NAME>.yaml` and `config/model/<NAME>.yaml`.

### 3. Testing / Inference
Evaluate a trained model, using its predictions to warm-start and locally-branch the solver:
```bash
python test.py dataset=CA cuda=0 model.delta=400          # Gurobi
python test.py dataset=CA cuda=0 solver=scip model.delta=200   # SCIP
```
`model.delta` is the local-branching neighborhood radius `Δ`.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{deng2026perturbation,
  title     = {A Perturbation-Augmented Unsupervised Learning Framework for Integer Linear Programming},
  author    = {Deng, Yufan and Pu, Tianle and Hu, Zhijing and Geng, Zijie and Zeng, Li and Hu, Xingchen and Huang, Kuihua and Wu, Junjie and Fan, Changjun},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2026},
  doi       = {10.1145/3770855.3817737}
}
```
