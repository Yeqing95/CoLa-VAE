# ============================================================
# run_CoLa_VAE.py
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scipy
import random
from CoLa_VAE import COLAVAE
from preprocess import normalize, geneSelection

from ccc_methods.compute_ccc_cellchat import build_ccc_laplacian_from_expr as build_ccc_cellchat
from ccc_methods.compute_ccc_cellphonedb import build_ccc_laplacian_from_expr as build_ccc_cellphonedb
from ccc_methods.compute_ccc_italk import build_ccc_laplacian_from_expr as build_ccc_italk
from ccc_methods.compute_ccc_cytotalk import build_ccc_laplacian_from_expr as build_ccc_cytotalk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# argparse
# ============================================================

parser = argparse.ArgumentParser(
    description='Communication Laplacian Variational Autoencoder (CoLa-VAE) Runner',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input', required=True, help='input .h5ad file')
parser.add_argument('--output_prefix', default='output', help='output prefix')

# HVG selection: 0 => use all genes; >0 => pick top n HVGs (loss only)
parser.add_argument('--select_genes', type=int, default=0, help='Number of HVGs for loss mask (0=all genes)')

# CCC method
parser.add_argument(
    '--ccc_method',
    type=str,
    default='cellchat',
    choices=['cellchat', 'cellphonedb', 'italk', 'cytotalk'],
    help='Method used to compute CCC Laplacian'
)

# latent dims
parser.add_argument('--CCC_dim', type=int, default=4)
parser.add_argument('--Normal_dim', type=int, default=16)

# VAE architecture
parser.add_argument('--encoder_layers', nargs='+', type=int, default=[256, 128])
parser.add_argument('--decoder_layers', nargs='+', type=int, default=[256])

parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--ccc_update_interval', type=int, default=10)

parser.add_argument('--init_beta', type=float, default=1)
parser.add_argument('--min_beta', type=float, default=0.01)
parser.add_argument('--max_beta', type=float, default=10)
parser.add_argument('--KL_loss', type=float, default=0.5)

# λ schedule
parser.add_argument('--lambda_start', type=float, default=0.2)
parser.add_argument('--lambda_max', type=float, default=1.0)
parser.add_argument('--lambda_step', type=float, default=0.2)
parser.add_argument('--lambda_update_interval', type=int, default=10)

# training
parser.add_argument('--batch_size', default=128)
parser.add_argument('--maxiter', type=int, default=500)
parser.add_argument('--train_size', type=float, default=0.95)
parser.add_argument('--patience', type=int, default=50)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)

parser.add_argument('--noise', type=float, default=0.0)
parser.add_argument('--dropoutE', type=float, default=0.0)
parser.add_argument('--dropoutD', type=float, default=0.0)

parser.add_argument('--device', default='cuda')

# CCC computation
parser.add_argument('--ccc_cutoff', type=float, default=0.1)
parser.add_argument('--ccc_top_k', type=int, default=100)
parser.add_argument('--ccc_alpha', type=float, default=0.5)
parser.add_argument('--ccc_Kh', type=float, default=1.0)
parser.add_argument('--ccc_batch_size_pairs', type=int, default=4)
parser.add_argument('--ccc_blockwise', action='store_true', help='Use blockwise CCC computation for large datasets')
parser.add_argument('--ccc_cell_block_size', type=int, default=1024, help='Cell block size for blockwise CCC computation')
# latent saving
parser.add_argument('--latent_interval', type=int, default=None, help='Interval to save intermediate latent (None=off)')

# LR scheduler
parser.add_argument('--use_scheduler', action='store_true', help='Enable learning rate scheduler')
parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'step', 'plateau'])
parser.add_argument('--scheduler_step', type=int, default=10)
parser.add_argument('--scheduler_gamma', type=float, default=0.5)

# Random seed
parser.add_argument('--seed', type=int, default=2026, help='Random seed')

args = parser.parse_args()
set_seed(args.seed)

# ============================================================
# Load h5ad
# ============================================================

print("\nReading input:", args.input)
adata = sc.read_h5ad(args.input)

# If raw counts are stored in adata.raw, move them back to adata.X
if adata.raw is not None:
    print("Detected adata.raw.X, replacing adata.X with raw counts.")
    adata.X = adata.raw.X.copy()
else:
    print("No adata.raw found, using adata.X as input.")

if adata.X.dtype != np.float64:
    adata.X = adata.X.astype(np.float64)

# ============================================================
# Normalize
# ============================================================

adata = normalize(
    adata,
    size_factors=True,
    normalize_input=True,
    logtrans_input=True
)

n_cells, n_genes = adata.X.shape
print("Data shape after preprocess:", adata.shape)

# ============================================================
# HVG selection
# ============================================================

if args.select_genes > 0:
    print(f"Selecting top {args.select_genes} HVGs (for loss mask only)...")

    raw_mat = adata.raw.X
    if scipy.sparse.issparse(raw_mat):
        data_for_hvg = raw_mat
    else:
        data_for_hvg = np.asarray(raw_mat)

    selected_mask = geneSelection(
        data_for_hvg,
        n=args.select_genes,
        plot=False,
        verbose=1
    )

    n_selected = int(selected_mask.sum())
    print(f"HVG selection picked {n_selected} genes.")

    if n_selected < 50:
        print("⚠️ WARNING: HVG selection returned < 50 genes. "
              "Fallback to using all genes for loss.")
        hvg_idx = np.arange(n_genes, dtype=int)
    else:
        hvg_idx = np.where(selected_mask)[0]
else:
    print("No HVG selection: using all genes for NB loss.")
    hvg_idx = np.arange(n_genes, dtype=int)

# ============================================================
# Decide batch size
# ============================================================

if args.batch_size == "auto":
    if n_cells <= 4096:
        batch_size = 128
    elif n_cells <= 8192:
        batch_size = 256
    else:
        batch_size = 512
else:
    batch_size = int(args.batch_size)

print("Using batch_size:", batch_size)


# ============================================================
# Choose CCC builder
# ============================================================

if args.ccc_method == "cellchat":
    ccc_builder = build_ccc_cellchat
elif args.ccc_method == "cellphonedb":
    ccc_builder = build_ccc_cellphonedb
elif args.ccc_method == "italk":
    ccc_builder = build_ccc_italk
elif args.ccc_method == "cytotalk":
    ccc_builder = build_ccc_cytotalk
else:
    raise ValueError(f"Unknown ccc_method: {args.ccc_method}")

print(f"Using CCC method: {args.ccc_method}")


# ============================================================
# Build COLAVAE model
# ============================================================

print("\nBuilding COLAVAE model...")

pos_index = np.arange(n_cells).astype(int)
n_genes_full = n_genes

model = COLAVAE(
    adata=adata,
    input_dim=n_genes_full,
    CCC_dim=args.CCC_dim,
    Normal_dim=args.Normal_dim,
    encoder_layers=args.encoder_layers,
    decoder_layers=args.decoder_layers,
    noise=args.noise,
    encoder_dropout=args.dropoutE,
    decoder_dropout=args.dropoutD,
    KL_loss=args.KL_loss,
    dynamicVAE=True,
    init_beta=args.init_beta,
    min_beta=args.min_beta,
    max_beta=args.max_beta,

    # HVG mask
    hvg_idx=hvg_idx,

    # CCC computation config
    ccc_builder=ccc_builder,
    ccc_cutoff=args.ccc_cutoff,
    ccc_top_k=args.ccc_top_k,
    ccc_alpha=args.ccc_alpha,
    ccc_Kh=args.ccc_Kh,
    ccc_batch_size_pairs=args.ccc_batch_size_pairs,
    ccc_blockwise=args.ccc_blockwise,
    ccc_cell_block_size=args.ccc_cell_block_size,

    warmup_epochs=args.warmup_epochs,
    ccc_update_interval=args.ccc_update_interval,

    laplacian_lambda_start=args.lambda_start,
    laplacian_lambda_max=args.lambda_max,
    laplacian_lambda_step=args.lambda_step,
    laplacian_lambda_update_interval=args.lambda_update_interval,

    dtype=torch.float64,
    device=args.device
)

print(model)


# ============================================================
# Prepare raw counts & size factors
# ============================================================

raw_counts = adata.raw.X if adata.raw is not None else adata.X

if scipy.sparse.issparse(raw_counts):
    raw_counts = raw_counts.toarray()
if scipy.sparse.issparse(adata.X):
    ncounts = adata.X.toarray()
else:
    ncounts = adata.X

raw_counts = raw_counts.astype(np.float64)
ncounts = ncounts.astype(np.float64)

size_factors = adata.obs.size_factors.values.astype(np.float64)


# ============================================================
# Train model
# ============================================================

print("\nStart training COLAVAE...\n")

model.train_model(
    pos=pos_index,
    ncounts=ncounts,
    raw_counts=raw_counts,
    size_factors=size_factors,
    lr=args.lr,
    weight_decay=args.weight_decay,
    batch_size=batch_size,
    num_samples=1,
    train_size=args.train_size,
    maxiter=args.maxiter,
    patience=args.patience,
    model_weights=args.output_prefix + "_model.pt",
    save_latent_interval=args.latent_interval,
    latent_prefix="latent",
    output_prefix=args.output_prefix,
    use_scheduler=args.use_scheduler,
    scheduler_type=args.scheduler_type,
    scheduler_step=args.scheduler_step,
    scheduler_gamma=args.scheduler_gamma,
    seed=args.seed,
)


# ============================================================
# Save final latent embedding (with cell names)
# ============================================================

print("\nSaving final latent embedding...")

latent = model.batching_latent_samples(
    X=pos_index,
    Y=ncounts,
    batch_size=batch_size
)

latent_df = pd.DataFrame(
    latent,
    index=adata.obs_names,
)
latent_df.to_csv(args.output_prefix + "_latent.csv")

# ============================================================
# Save full-gene denoised counts (with gene & cell names)
# ============================================================

print("Saving denoised counts...")

denoised = model.batching_denoise_counts(
    X=pos_index,
    Y=ncounts,
    n_samples=25,
    batch_size=batch_size
)

denoised_df = pd.DataFrame(
    denoised,
    index=adata.obs_names,
    columns=adata.var_names
)
denoised_df.to_csv(args.output_prefix + "_denoised.csv")

print("\nAll done.\n")


