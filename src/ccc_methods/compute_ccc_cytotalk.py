# ============================================================
# compute_ccc_cytotalk.py (Optimized Version)
# CytoTalk single-cell CCC → BI Frobenius distance → Laplacian
# Only MI for LR pairs (500x faster)
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


# ============================================================
# Device / dtype helpers (same as other CCC files)
# ============================================================

def _select_device(device):
    if isinstance(device, torch.device):
        return device
    dev = str(device).lower()
    if dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype(device, dtype=None):
    if dtype is not None:
        return torch.float64 if "64" in str(dtype).lower() else torch.float32
    return torch.float64 if device.type == "cuda" else torch.float32


# ============================================================
# Load CytoTalk LR DB
# ============================================================

def _load_cytotalk_lr_db():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "CytoTalk_LR_database.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"CytoTalk_LR_database.csv not found at:\n{path}"
        )

    df = pd.read_csv(path)
    if not {"ligand", "receptor"}.issubset(df.columns):
        raise ValueError("LR DB must contain columns: ligand, receptor")

    return df


# ============================================================
# PEM for single-cell CytoTalk
# ============================================================

def _compute_PEM_singlecell(expr_np):
    """
    PEM[i,A] = log10( expr[i,A] / expected[i,A] )
    expected = global_gene_sum * (library_size / total_library)
    """
    X = np.asarray(expr_np, float)
    n_cells, n_genes = X.shape

    G = X.sum(axis=0)         # gene total
    s = X.sum(axis=1)         # cell library size
    S = s.sum()

    expected = np.outer(s / S, G)
    eps = 1e-12

    PEM = np.log10(np.maximum(X, eps) / np.maximum(expected, eps))
    PEM[PEM < 0] = 0
    return PEM


# ============================================================
# Optimized MI calculation — only LR genes!
# ============================================================

def _compute_MI_for_pairs(X_disc, lr_pairs_idx):
    """
    X_disc: discretized expr, shape (cells × genes)
    lr_pairs_idx: list of (i,j) indices for LR pairs

    Returns MI_dict[(i,j)] = MI(i,j)
    """
    MI_dict = {}

    for i, j in lr_pairs_idx:
        MI_ij = mutual_info_score(X_disc[:, i], X_disc[:, j])
        MI_ji = mutual_info_score(X_disc[:, j], X_disc[:, i])
        MI_dict[(i, j)] = (MI_ij, MI_ji)

    return MI_dict


# ============================================================
# Prepare LR items with optimized MI / PEM
# ============================================================

def _prepare_LR_items_cytotalk(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    device=None,
    dtype=None,
    verbose=True,
):
    device = _select_device(device)
    dtype  = _select_dtype(device, dtype)

    expr_np = expr if isinstance(expr, np.ndarray) else expr.detach().cpu().numpy()
    expr_np = np.asarray(expr_np, dtype=float)

    n_cells, n_genes = expr_np.shape
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Load LR DB
    df = _load_cytotalk_lr_db()

    # Collect LR index pairs
    lr_pairs = []
    for _, row in df.iterrows():
        L = row["ligand"]
        R = row["receptor"]
        if L in gene_to_idx and R in gene_to_idx:
            lr_pairs.append((gene_to_idx[L], gene_to_idx[R]))

    # PEM
    if verbose:
        print("Computing PEM...")
    PEM = _compute_PEM_singlecell(expr_np)

    # Discretize for MI
    if verbose:
        print("Discretizing expression for MI...")
    X_disc = np.zeros_like(expr_np, dtype=int)
    for g in range(n_genes):
        edges = np.histogram(expr_np[:, g], bins=10)[1]
        X_disc[:, g] = np.digitize(expr_np[:, g], edges)

    # Optimized MI: only LR pairs
    if verbose:
        print("Computing MI for LR pairs ONLY...")
    MI_dict = _compute_MI_for_pairs(X_disc, lr_pairs)

    # Build LR_items
    LR_items = []
    eps = 1e-12

    for (i, j) in lr_pairs:
        pem_L = PEM[:, i]
        pem_R = PEM[:, j]

        expr_score = (pem_L.mean() + pem_R.mean()) / 2
        if expr_score < cutoff:
            continue

        MI_ij, MI_ji = MI_dict[(i, j)]
        H_i = mutual_info_score(X_disc[:, i], X_disc[:, i])
        H_j = mutual_info_score(X_disc[:, j], X_disc[:, j])
        H_min = max(H_i, H_j)

        item = {
            "i": i,
            "j": j,
            "PEM_L": pem_L,
            "PEM_R": pem_R,
            "MI_ij": MI_ij,
            "MI_ji": MI_ji,
            "H_min": H_min,
            "score": expr_score,
        }
        LR_items.append(item)

    LR_items = sorted(LR_items, key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        LR_items = LR_items[:top_k]

    return LR_items, n_cells, device, dtype


# ============================================================
# CytoTalk → BI Frobenius distance
# ============================================================

def compute_ccc_bi_distance_from_expr(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    alpha=0.5,
    Kh=1.0,
    device="cuda",
    dtype=None,
    batch_size_pairs=10,
    verbose=True,
):
    LR_items, n_cells, device_t, dtype_t = _prepare_LR_items_cytotalk(
        expr,
        genes,
        top_k=top_k,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )

    D_out2 = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)
    D_in2  = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)

    eps = 1e-12

    for start in range(0, len(LR_items), batch_size_pairs):
        batch = LR_items[start: start + batch_size_pairs]
        P_list = []

        for item in batch:
            i, j = item["i"], item["j"]
            pem_L = item["PEM_L"][:, None]      # (cells,1)
            pem_R = item["PEM_R"][None, :]      # (1,cells)

            expr_score = (pem_L + pem_R) / 2

            MI_ij, MI_ji = item["MI_ij"], item["MI_ji"]
            H_min = item["H_min"]
            nonself_A = -np.log10((MI_ij + eps) / (H_min + eps))
            nonself_B = -np.log10((MI_ji + eps) / (H_min + eps))
            nonself = (nonself_A + nonself_B) / 2

            P_list.append(torch.tensor(expr_score * nonself, device=device_t, dtype=dtype_t))

        P = torch.stack(P_list, dim=0)  # (B, cells, cells)

        # OUT profile
        X_out = P.reshape(-1, n_cells)
        Gram_out = X_out.T @ X_out
        diag_out = torch.diag(Gram_out)
        D_out2 += diag_out[:, None] + diag_out[None, :] - 2*Gram_out

        # IN profile
        X_in = P.permute(1, 0, 2).reshape(n_cells, -1)
        Gram_in = X_in @ X_in.T
        diag_in = torch.diag(Gram_in)
        D_in2 += diag_in[:, None] + diag_in[None, :] - 2*Gram_in

        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    # BI aggregation
    D_bi2 = alpha * D_out2 + (1-alpha) * D_in2
    D_bi = torch.sqrt(torch.clamp(D_bi2, min=0) + eps)

    D_np = D_bi.detach().cpu().numpy()
    D_np = 0.5*(D_np + D_np.T)
    np.fill_diagonal(D_np, 0)

    return D_np


# ============================================================
# Laplacian builder
# ============================================================

def sparsify_topk(S, k=30):
    S = S.copy()
    n = S.shape[0]
    for i in range(n):
        idx = np.argsort(S[i])[:-k-1]
        S[i, idx] = 0.0
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 0.0)
    return S

def build_laplacian_from_distance(D_bi):
    D = np.asarray(D_bi, dtype=np.float64)

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0
    finite_mask = np.isfinite(D)
    if not finite_mask.all():
        finite_max = np.max(D[finite_mask])
        D[~finite_mask] = finite_max

    pos = D[D > 0]
    if pos.size == 0:
        n = D.shape[0]
        return np.zeros((n, n), dtype=np.float64)

    sigma = np.median(pos)
    if sigma <= 0:
        sigma = np.mean(pos) if np.mean(pos) > 0 else 1.0

    S = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    S = 0.5 * (S + S.T)
    S[S < 0] = 0.0
    np.fill_diagonal(S, 0.0)

    S = sparsify_topk(S, k=30)
    deg = S.sum(axis=1)
    deg[deg <= 1e-12] = 1e-12
    D_inv_sqrt = 1.0 / np.sqrt(deg)

    n = S.shape[0]
    I = np.eye(n, dtype=np.float64)
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    L = I - D_inv_sqrt_mat @ S @ D_inv_sqrt_mat

    L = 0.5 * (L + L.T)

    return L


# ============================================================
# Top-level API for cccVAE
# ============================================================

def build_ccc_laplacian_from_expr(
    expr,
    genes,
    cutoff=0.1,
    top_k=None,
    alpha=0.5,
    Kh=1.0,
    device="cuda",
    dtype=None,
    batch_size_pairs=4,
    blockwise=False,
    cell_block_size=1024,
    verbose=True,
):
    D = compute_ccc_bi_distance_from_expr(
        expr,
        genes,
        top_k=top_k,
        cutoff=cutoff,
        alpha=alpha,
        Kh=Kh,
        device=device,
        dtype=dtype,
        batch_size_pairs=batch_size_pairs,
        verbose=verbose,
    )
    return build_laplacian_from_distance(D)


