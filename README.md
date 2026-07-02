# CoLa-VAE

**CoLa-VAE** is a communication-guided variational autoencoder for single-cell and spatial transcriptomic analysis. The model jointly learns a latent representation and a denoised expression matrix by incorporating ligand–receptor-derived cell–cell communication (CCC) topology through dynamic graph Laplacian regularization.

CoLa-VAE can be used to generate:

* CCC-aware latent embeddings for visualization and clustering
* denoised expression matrices for downstream analysis
* communication-guided representations using different CCC scoring strategies
* spatially constrained CoLa-VAE outputs when spatial information is available

## Installation

CoLa-VAE was developed and tested with Python 3.10 and GPU-enabled PyTorch.

### 1. Clone the repository

```bash
git clone https://github.com/Yeqing95/CoLa-VAE.git
cd CoLa-VAE
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate colavae
```

A minimal `environment.yml` is:

```yaml
name: colavae
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - numpy==2.2.6
      - pandas==2.3.3
      - scipy==1.15.3
      - scanpy
      - anndata
      - scikit-learn
      - numba==0.62.1
      - dask==2025.11.0
      - distributed==2025.11.0
      - matplotlib
      - tqdm
      - tensorboard==2.20.0
      - tensorflow==2.20.0
      - keras==3.12.0
      - protobuf==6.33.2
      - --extra-index-url https://download.pytorch.org/whl/cu118
      - torch==2.7.1+cu118
      - torchvision==0.22.1+cu118
      - torchaudio==2.7.1+cu118
```

Alternatively, the environment can be created manually:

```bash
conda create -n colavae python=3.10 pip -y
conda activate colavae

pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.3
pip install scanpy anndata scikit-learn numba==0.62.1
pip install dask==2025.11.0 distributed==2025.11.0 matplotlib tqdm
pip install tensorboard==2.20.0 tensorflow==2.20.0 keras==3.12.0 protobuf==6.33.2
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify GPU installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If GPU support is correctly configured, the command should return `True` for CUDA availability.

## Input data

CoLa-VAE takes an AnnData `.h5ad` file as input.

The input file should contain a gene expression matrix with cells as rows and genes as columns:

```text
adata.X: cells × genes expression matrix
adata.obs: cell-level metadata
adata.var: gene-level metadata
```

The expression matrix can be raw counts or preprocessed counts depending on the analysis design. For denoising and count-based modeling, raw or lightly filtered count matrices are recommended.

## Quick start

To run CoLa-VAE on a single-cell dataset:

```bash
python run_CoLa_VAE.py \
  --input data/pbmc3k.h5ad \
  --output_prefix results/pbmc3k_colavae \
  --ccc_method cellchat \
  --CCC_dim 4 \
  --Normal_dim 16 \
  --select_genes 2000 \
  --maxiter 500 \
  --batch_size 128 \
  --device cuda
```

This command runs CoLa-VAE using the CellChat-like CCC scoring module, with 4 CCC-aware latent dimensions and 16 expression-related latent dimensions.

## Choosing the CCC module

CoLa-VAE supports multiple ligand–receptor-based CCC scoring strategies:

```bash
--ccc_method cellchat
--ccc_method cellphonedb
--ccc_method italk
--ccc_method cytotalk
```

For example, to use the CellPhoneDB-like CCC module:

```bash
python run_CoLa_VAE.py \
  --input data/pbmc3k.h5ad \
  --output_prefix results/pbmc3k_cellphonedb \
  --ccc_method cellphonedb \
  --CCC_dim 4 \
  --Normal_dim 16 \
  --select_genes 2000 \
  --device cuda
```

## Main parameters

### Input and output

| Argument          | Description             | Default  |
| ----------------- | ----------------------- | -------- |
| `--input`         | Input `.h5ad` file      | required |
| `--output_prefix` | Prefix for output files | `output` |

### Gene selection

| Argument         | Description                                                                       | Default |
| ---------------- | --------------------------------------------------------------------------------- | ------- |
| `--select_genes` | Number of highly variable genes used for the loss mask. Use `0` to use all genes. | `0`     |

### CCC module

| Argument       | Description                                                                  | Default    |
| -------------- | ---------------------------------------------------------------------------- | ---------- |
| `--ccc_method` | CCC scoring method. Options: `cellchat`, `cellphonedb`, `italk`, `cytotalk`. | `cellchat` |
| `--ccc_cutoff` | Cutoff used in CCC graph construction.                                       | `0.1`      |
| `--ccc_top_k`  | Number of top CCC neighbors retained.                                        | `100`      |
| `--ccc_alpha`  | Weight balancing outgoing and incoming CCC profiles.                         | `0.5`      |
| `--ccc_Kh`     | Hill-function saturation parameter used by the CellChat-like module.         | `1.0`      |

### Latent dimensions

| Argument       | Description                                     | Default |
| -------------- | ----------------------------------------------- | ------- |
| `--CCC_dim`    | Number of CCC-aware latent dimensions.          | `4`     |
| `--Normal_dim` | Number of expression-related latent dimensions. | `16`    |

### Model architecture

| Argument           | Description                 | Default   |
| ------------------ | --------------------------- | --------- |
| `--encoder_layers` | Encoder hidden layer sizes. | `256 128` |
| `--decoder_layers` | Decoder hidden layer sizes. | `256`     |

Example:

```bash
python run_CoLa_VAE.py \
  --input data/pbmc3k.h5ad \
  --output_prefix results/pbmc3k_deeper \
  --encoder_layers 512 256 128 \
  --decoder_layers 256 512
```

### Training

| Argument         | Description                                    | Default |
| ---------------- | ---------------------------------------------- | ------- |
| `--maxiter`      | Maximum number of training epochs.             | `500`   |
| `--batch_size`   | Batch size.                                    | `128`   |
| `--train_size`   | Fraction of cells used for training.           | `0.95`  |
| `--patience`     | Early stopping patience.                       | `50`    |
| `--lr`           | Learning rate.                                 | `1e-3`  |
| `--weight_decay` | Weight decay.                                  | `1e-6`  |
| `--device`       | Device used for training. Use `cuda` or `cpu`. | `cuda`  |
| `--seed`         | Random seed.                                   | `2026`  |

### Dynamic CCC update

| Argument                   | Description                                                        | Default |
| -------------------------- | ------------------------------------------------------------------ | ------- |
| `--warmup_epochs`          | Number of warm-up epochs before applying CCC graph regularization. | `10`    |
| `--ccc_update_interval`    | Number of epochs between CCC graph updates.                        | `10`    |
| `--lambda_start`           | Initial CCC graph regularization weight.                           | `0.2`   |
| `--lambda_max`             | Maximum CCC graph regularization weight.                           | `1.0`   |
| `--lambda_step`            | Step size for increasing graph regularization weight.              | `0.2`   |
| `--lambda_update_interval` | Number of epochs between lambda updates.                           | `10`    |

### Large datasets

For large datasets, blockwise CCC computation can be enabled to reduce memory usage:

```bash
python run_CoLa_VAE.py \
  --input data/large_dataset.h5ad \
  --output_prefix results/large_dataset_colavae \
  --ccc_method cellchat \
  --ccc_blockwise \
  --ccc_cell_block_size 1024 \
  --device cuda
```

The `--ccc_cell_block_size` parameter controls the number of cells processed per block during CCC graph construction.

### Learning rate scheduler

A learning rate scheduler can be enabled using:

```bash
--use_scheduler
```

Supported scheduler types are:

```bash
--scheduler_type cosine
--scheduler_type step
--scheduler_type plateau
```

Example:

```bash
python run_CoLa_VAE.py \
  --input data/pbmc3k.h5ad \
  --output_prefix results/pbmc3k_scheduler \
  --ccc_method cellchat \
  --use_scheduler \
  --scheduler_type cosine
```

## Output

CoLa-VAE saves outputs using the user-provided `--output_prefix`.

Typical outputs include:

```text
<output_prefix>_latent.csv or .npy
<output_prefix>_denoised.csv or .h5ad
<output_prefix>_model.pt
<output_prefix>_training_log.csv
```

The latent representation can be used for downstream UMAP visualization, clustering, and batch comparison. The denoised expression matrix can be used for downstream analyses such as differential expression, pathway analysis, cell–cell communication inference, and spatial deconvolution.

## Contact

For questions or issues, please contact:

```text
Yeqing Chen
Department of Computer Science
New Jersey Institute of Technology
GitHub: https://github.com/Yeqing95/CoLa-VAE
```
