# ============================================================
# run_CoLa_VAE_spatial.py
# Spatial-aware CoLa-VAE runner for 10x Visium output folder
#
# This version uses only compute_ccc_cellchat_spatial.py.
# No separate graph_loss_spatial is added.
# ============================================================

import os
import argparse
import json
import random

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import torch

from CoLa_VAE_spatial import COLAVAE_Spatial
from preprocess import normalize, geneSelection
from ccc_methods.compute_ccc_cellchat_spatial import build_ccc_laplacian_from_expr as build_ccc_cellchat_spatial


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Auto-detect Visium files
# ============================================================

def find_visium_files(data_dir):
    data_dir = os.path.abspath(data_dir)

    matrix_h5 = None
    for fn in ["filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix.h5.gz"]:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            matrix_h5 = fp
            break

    if matrix_h5 is None:
        raise FileNotFoundError("Cannot find filtered_feature_bc_matrix.h5 or filtered_feature_bc_matrix.h5.gz")

    spatial_dir = os.path.join(data_dir, "spatial")
    if not os.path.exists(spatial_dir):
        raise FileNotFoundError("Cannot find spatial/ directory")

    pos_file = None
    for fn in ["tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions_list.txt"]:
        fp = os.path.join(spatial_dir, fn)
        if os.path.exists(fp):
            pos_file = fp
            break

    parquet_fp = os.path.join(spatial_dir, "tissue_positions.parquet")
    if pos_file is None and os.path.exists(parquet_fp):
        pos_file = parquet_fp

    if pos_file is None:
        raise FileNotFoundError(
            "Cannot find tissue_positions.csv, tissue_positions_list.csv, "
            "tissue_positions_list.txt, or tissue_positions.parquet"
        )

    scale_file = os.path.join(spatial_dir, "scalefactors_json.json")
    if not os.path.exists(scale_file):
        print("WARNING: scalefactors_json.json not found. Using raw pixel coordinates.")
        scale_file = None

    return matrix_h5, pos_file, scale_file


def load_visium_positions(pos_file, obs_names):
    """
    Load and align Visium spatial positions to adata.obs_names.
    Supports Space Ranger old and new formats.
    """
    if pos_file.endswith(".parquet"):
        pos_df = pd.read_parquet(pos_file)
    else:
        # Some Space Ranger files have headers; old tissue_positions_list has no header.
        tmp = pd.read_csv(pos_file, sep=",", header=None)
        if tmp.shape[1] == 6:
            first = str(tmp.iloc[0, 0]).lower()
            if first in ["barcode", "barcodes"]:
                tmp = pd.read_csv(pos_file, sep=",", header=0)
            else:
                tmp.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        pos_df = tmp

    # Standardize column names
    rename_map = {
        "barcodes": "barcode",
        "pxl_row_in_fullres": "pxl_row",
        "pxl_col_in_fullres": "pxl_col",
    }
    pos_df = pos_df.rename(columns=rename_map)

    required = {"barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"}
    missing = required - set(pos_df.columns)
    if missing:
        raise ValueError(f"Position file is missing columns: {missing}. Found columns: {list(pos_df.columns)}")

    pos_df = pos_df.set_index("barcode")
    pos_df = pos_df.loc[obs_names]
    return pos_df


# ============================================================
# argparse
# ============================================================

parser = argparse.ArgumentParser(
    description="Spatial-aware CoLa-VAE Runner: spatial-masked CellChat CCC graph",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--data_dir", required=True, help="10x Visium output folder")
parser.add_argument("--output_prefix", default="colavae_spatial", help="Output prefix")

# HVG selection: 0 => use all genes; >0 => pick top n HVGs for NB loss mask only
parser.add_argument("--select_genes", type=int, default=0, help="Number of HVGs for loss mask; 0 = all genes")

# latent dims
parser.add_argument("--CCC_dim", type=int, default=4)
parser.add_argument("--Normal_dim", type=int, default=16)

# VAE architecture
parser.add_argument("--encoder_layers", nargs="+", type=int, default=[256, 128])
parser.add_argument("--decoder_layers", nargs="+", type=int, default=[256])

# Warmup and CCC schedule
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--ccc_update_interval", type=int, default=10)

parser.add_argument("--init_beta", type=float, default=1.0)
parser.add_argument("--min_beta", type=float, default=0.01)
parser.add_argument("--max_beta", type=float, default=10.0)
parser.add_argument("--KL_loss", type=float, default=0.5)

# CCC graph lambda schedule
parser.add_argument("--lambda_start", type=float, default=0.2)
parser.add_argument("--lambda_max", type=float, default=1.0)
parser.add_argument("--lambda_step", type=float, default=0.2)
parser.add_argument("--lambda_update_interval", type=int, default=10)

# training
parser.add_argument("--batch_size", default="128", help="Batch size or auto")
parser.add_argument("--maxiter", type=int, default=500)
parser.add_argument("--train_size", type=float, default=0.95)
parser.add_argument("--patience", type=int, default=50)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-6)

parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--dropoutE", type=float, default=0.0)
parser.add_argument("--dropoutD", type=float, default=0.0)

parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])

# CellChat CCC computation
parser.add_argument("--ccc_cutoff", type=float, default=0.1)
parser.add_argument("--ccc_top_k", type=int, default=100)
parser.add_argument("--ccc_alpha", type=float, default=0.5)
parser.add_argument("--ccc_Kh", type=float, default=1.0)
parser.add_argument("--ccc_batch_size_pairs", type=int, default=4)
parser.add_argument("--ccc_blockwise", action="store_true", help="Use blockwise CCC computation for large datasets")
parser.add_argument("--ccc_cell_block_size", type=int, default=1024)

# Spatial mask parameters for CCC pruning
parser.add_argument("--spatial_radius_cutoff", type=float, default=100.0)
parser.add_argument("--spatial_scale", type=float, default=50.0)
parser.add_argument("--spatial_mask_mode", type=str, default="exp", choices=["exp", "linear", "binary"])

# Latent saving and scheduler
parser.add_argument("--latent_interval", type=int, default=None, help="Interval to save intermediate latent; None = off")
parser.add_argument("--use_scheduler", action="store_true", help="Enable learning rate scheduler")
parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["cosine", "step", "plateau"])
parser.add_argument("--scheduler_step", type=int, default=10)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)

# random seed
parser.add_argument("--seed", type=int, default=2026)

args = parser.parse_args()
set_seed(args.seed)

if args.device == "cuda" and not torch.cuda.is_available():
    print("WARNING: CUDA requested but not available. Switching to CPU.")
    args.device = "cpu"

if args.dtype == "float64":
    dtype_torch = torch.float64
    np_dtype = np.float64
else:
    dtype_torch = torch.float32
    np_dtype = np.float32


# ============================================================
# 1. Find and load Visium files
# ============================================================

matrix_h5, pos_file, scale_file = find_visium_files(args.data_dir)

print("Matrix:", matrix_h5)
print("Positions:", pos_file)
print("Scalefactors:", scale_file)

adata = sc.read_10x_h5(matrix_h5)
adata.var_names_make_unique()
print("Loaded matrix:", adata.shape)


# ============================================================
# 2. Load and align spatial coordinates
# ============================================================

pos_df = load_visium_positions(pos_file, adata.obs_names)

if scale_file is not None:
    with open(scale_file, "r") as f:
        scale_json = json.load(f)
    # full-resolution pixel coords are already fullres in newer Space Ranger files.
    # Using tissue_lowres_scalef keeps behavior consistent with your previous script.
    scale = scale_json.get("tissue_lowres_scalef", 1.0)
    print("Using tissue_lowres_scalef:", scale)
else:
    scale = 1.0
    print("No scalefactor found, using raw pixel coords.")

adata.obs["x"] = pos_df["pxl_col"].values.astype(np.float64) * scale
adata.obs["y"] = pos_df["pxl_row"].values.astype(np.float64) * scale
coords = adata.obs[["x", "y"]].values.astype(np.float64)


# ============================================================
# 3. Normalize
# ============================================================

adata = normalize(
    adata,
    filter_min_counts=False,
    size_factors=True,
    normalize_input=True,
    logtrans_input=True,
)

n_cells, n_genes = adata.shape
print("Data shape after preprocess:", adata.shape)


# ============================================================
# 4. HVG selection for NB loss mask only
# ============================================================

raw_mat = adata.raw.X if adata.raw is not None else adata.X
if scipy.sparse.issparse(raw_mat):
    raw_for_hvg = raw_mat
else:
    raw_for_hvg = np.asarray(raw_mat)

if args.select_genes > 0:
    print(f"Selecting top {args.select_genes} HVGs for NB loss mask only...")
    selected_mask = geneSelection(raw_for_hvg, n=args.select_genes, plot=False, verbose=1)
    n_selected = int(selected_mask.sum())
    print(f"HVG selection picked {n_selected} genes.")

    if n_selected < 50:
        print("WARNING: HVG selection returned < 50 genes. Falling back to all genes for loss.")
        hvg_idx = np.arange(n_genes, dtype=int)
    else:
        hvg_idx = np.where(selected_mask)[0]
else:
    print("No HVG selection: using all genes for NB loss.")
    hvg_idx = np.arange(n_genes, dtype=int)


# ============================================================
# 5. Decide batch size
# ============================================================

if str(args.batch_size).lower() == "auto":
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
# 6. Prepare matrices
# ============================================================

raw_counts = adata.raw.X if adata.raw is not None else adata.X
if scipy.sparse.issparse(raw_counts):
    raw_counts = raw_counts.toarray()
if scipy.sparse.issparse(adata.X):
    ncounts = adata.X.toarray()
else:
    ncounts = adata.X

raw_counts = raw_counts.astype(np_dtype)
ncounts = ncounts.astype(np_dtype)
size_factors = adata.obs.size_factors.values.astype(np_dtype)


# ============================================================
# 7. Build model
# ============================================================

print("\nBuilding spatial-aware CoLa-VAE model...")
pos_index = np.arange(n_cells).astype(int)

model = COLAVAE_Spatial(
    adata=adata,
    input_dim=n_genes,
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
    dtype=dtype_torch,
    device=args.device,

    hvg_idx=hvg_idx,

    ccc_builder=build_ccc_cellchat_spatial,
    ccc_cutoff=args.ccc_cutoff,
    ccc_top_k=args.ccc_top_k,
    ccc_alpha=args.ccc_alpha,
    ccc_Kh=args.ccc_Kh,
    ccc_batch_size_pairs=args.ccc_batch_size_pairs,
    ccc_blockwise=args.ccc_blockwise,
    ccc_cell_block_size=args.ccc_cell_block_size,

    spatial_coords=coords,
    spatial_radius_cutoff=args.spatial_radius_cutoff,
    spatial_scale=args.spatial_scale,
    spatial_mask_mode=args.spatial_mask_mode,

    warmup_epochs=args.warmup_epochs,
    ccc_update_interval=args.ccc_update_interval,

    laplacian_lambda_start=args.lambda_start,
    laplacian_lambda_max=args.lambda_max,
    laplacian_lambda_step=args.lambda_step,
    laplacian_lambda_update_interval=args.lambda_update_interval,
)

print(model)


# ============================================================
# 8. Train
# ============================================================

print("\nStart training spatial-aware CoLa-VAE...\n")

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
# 9. Save final latent, denoised counts, and coordinates
# ============================================================

print("\nSaving final latent embedding...")
latent = model.batching_latent_samples(
    X=pos_index,
    Y=ncounts,
    num_samples=1,
    batch_size=batch_size,
)
latent_df = pd.DataFrame(latent, index=adata.obs_names)
latent_df.to_csv(args.output_prefix + "_latent.csv")

print("Saving denoised counts...")
denoised = model.batching_denoise_counts(
    X=pos_index,
    Y=ncounts,
    n_samples=25,
    batch_size=batch_size,
)
denoised_df = pd.DataFrame(
    denoised,
    index=adata.obs_names,
    columns=adata.var_names,
)
denoised_df.to_csv(args.output_prefix + "_denoised.csv")

coord_df = adata.obs[["x", "y"]].copy()
coord_df.to_csv(args.output_prefix + "_spatial_coords.csv")

print("\nAll done.\n")
