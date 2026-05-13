# ============================================================
# compute_ccc_cellphonedb.py
# Auto-load LR DB from internal file (CellPhoneDB_interactors.csv)
# Single-cell CCC → BI distance → Laplacian
# ============================================================

import os
import torch
import numpy as np
import pandas as pd
import re


# ============================================================
# Device / dtype helpers
# ============================================================

def _select_device(device):
    if isinstance(device, torch.device):
        return device
    dev_str = str(device).lower()
    if dev_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if dev_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _select_dtype_for_device(device, dtype=None):
    if dtype is not None:
        if isinstance(dtype, torch.dtype):
            return dtype
        s = str(dtype).lower()
        if s in ("float64", "double"):
            return torch.float64
        return torch.float32

    if device.type == "cuda":
        return torch.float64
    elif device.type == "mps":
        return torch.float32
    else:
        return torch.float32


# ============================================================
# Utility functions
# ============================================================

def _parse_interactors(s):
    """
    Example:
        CDH1-ITGA2+ITGB1
        CEACAM5-CD1D
        COL10A1-ITGA10+ITGB1
    Returns:
        L_genes, R_genes
    """
    if pd.isna(s) or s == "":
        return [], []
    s = str(s).strip()
    if "-" not in s:
        return [], []
    lpart, rpart = s.split("-", 1)
    L = [x.strip() for x in lpart.split("+") if x.strip()]
    R = [x.strip() for x in rpart.split("+") if x.strip()]
    return L, R


def _min_complex_for_list(
    gene_list,
    gene_expr,
    n_cells,
    device,
    dtype,
    allow_missing=False,
):
    """
    CellPhoneDB v5 complex rule: expression = min(subunits)
    """
    if len(gene_list) == 0:
        return torch.zeros(n_cells, device=device, dtype=dtype)

    vecs = []
    for g in gene_list:
        if g in gene_expr:
            vecs.append(gene_expr[g])
        else:
            if allow_missing:
                return torch.zeros(n_cells, device=device, dtype=dtype)
            else:
                return None

    X = torch.stack(vecs, dim=0)
    return torch.min(X, dim=0).values



# ============================================================
# AUTO LOAD LR DATABASE (internal file)
# ============================================================

def _load_internal_lrdb():
    """
    Auto-load LR DB located in the same folder as this script.
    File must contain column: 'interactors'
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "CellPhoneDB_LR_database.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cannot find CellPhoneDB_LR_database.csv at:\n  {path}\n"
            f"Please put your LR DB file in this folder."
        )

    return pd.read_csv(path)



# ============================================================
# Prepare LR_items
# ============================================================

def _prepare_LR_items_cellphonedb(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    device=None,
    dtype=None,
    verbose=True,
):
    device = _select_device(device)
    dtype  = _select_dtype_for_device(device, dtype)

    # convert expression
    if isinstance(expr, np.ndarray):
        expr_t = torch.tensor(expr, device=device, dtype=dtype)
    else:
        expr_t = expr.to(device=device, dtype=dtype)

    n_cells, n_genes = expr_t.shape
    gene_expr = {g: expr_t[:, i] for i, g in enumerate(genes)}

    # auto load LR DB
    df = _load_internal_lrdb()
    if "interactors" not in df.columns:
        raise ValueError("CellPhoneDB_LR_database.csv must contain column 'interactors'.")

    LR_items = []

    for _, row in df.iterrows():
        L_genes, R_genes = _parse_interactors(row["interactors"])

        L_vals = _min_complex_for_list(L_genes, gene_expr, n_cells, device, dtype, allow_missing=False)
        if L_vals is None:
            continue

        R_vals = _min_complex_for_list(R_genes, gene_expr, n_cells, device, dtype, allow_missing=False)
        if R_vals is None:
            continue

        score = min(L_vals.mean().item(), R_vals.mean().item())
        if score < cutoff:
            continue

        LR_items.append(
            {
                "L_vals": L_vals,
                "R_vals": R_vals,
                "score":  score,
                "name":   row["interactors"],
            }
        )

    if len(LR_items) == 0:
        raise ValueError(f"No valid LR pairs after cutoff={cutoff}")

    LR_items = sorted(LR_items, key=lambda x: x["score"], reverse=True)
    if top_k is not None and top_k < len(LR_items):
        LR_items = LR_items[:top_k]

    return LR_items, n_cells, device, dtype



# ============================================================
# Compute BI Frobenius distance (CellChat-style API)
# ============================================================

def compute_ccc_bi_distance_from_expr(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    alpha=0.5,
    Kh=1.0,         # unused, kept for API compatibility
    device="cuda",
    dtype=None,
    batch_size_pairs=4,
    verbose=True,
):
    LR_items, n_cells, device_t, dtype_t = _prepare_LR_items_cellphonedb(
        expr=expr,
        genes=genes,
        top_k=top_k,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        verbose=verbose,
    )

    D_out2 = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)
    D_in2  = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)

    K = len(LR_items)

    for start in range(0, K, batch_size_pairs):
        end = min(start + batch_size_pairs, K)
        batch = LR_items[start:end]

        L_batch = torch.stack([item["L_vals"] for item in batch], dim=0).to(device_t, dtype_t)
        R_batch = torch.stack([item["R_vals"] for item in batch], dim=0).to(device_t, dtype_t)

        Li = L_batch[:, :, None]
        Rj = R_batch[:, None, :]
        P  = Li * Rj  # CellPhoneDB single-cell formula

        # OUT profile
        X_out = P.reshape(-1, n_cells)
        Gram_out = X_out.T @ X_out
        diag_out = torch.diag(Gram_out)
        contrib_out = diag_out[:, None] + diag_out[None, :] - 2 * Gram_out
        D_out2 += contrib_out

        # IN profile
        X_in = P.permute(1, 0, 2).reshape(n_cells, -1)
        Gram_in = X_in @ X_in.T
        diag_in = torch.diag(Gram_in)
        contrib_in = diag_in[:, None] + diag_in[None, :] - 2 * Gram_in
        D_in2 += contrib_in

        del L_batch, R_batch, Li, Rj, P
        del X_out, Gram_out, diag_out, contrib_out
        del X_in, Gram_in, diag_in, contrib_in

        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    eps = 1e-12
    D_out2 = torch.clamp(D_out2, min=0.0)
    D_in2  = torch.clamp(D_in2,  min=0.0)

    # CellChat-style BI aggregation
    D_bi2 = alpha * D_out2 + (1 - alpha) * D_in2
    D_bi2 = torch.clamp(D_bi2, min=0.0)
    D_bi  = torch.sqrt(D_bi2 + eps)

    # convert to numpy
    D_bi_np = D_bi.detach().cpu().numpy().astype(np.float64)
    D_bi_np = 0.5 * (D_bi_np + D_bi_np.T)
    np.fill_diagonal(D_bi_np, 0.0)

    finite = np.isfinite(D_bi_np)
    if not finite.all():
        max_val = np.max(D_bi_np[finite])
        D_bi_np[~finite] = max_val

    return D_bi_np



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
# Top-level API
# ============================================================

def build_ccc_laplacian_from_expr(
    expr,
    genes,
    cutoff=0.1,
    top_k=None,
    alpha=0.5,
    Kh=1.0,   # unused but kept for compatibility
    device="cuda",
    dtype=None,
    batch_size_pairs=4,
    blockwise=False,
    cell_block_size=1024,
    verbose=True,
):
    D_bi = compute_ccc_bi_distance_from_expr(
        expr=expr,
        genes=genes,
        top_k=top_k,
        cutoff=cutoff,
        alpha=alpha,
        Kh=Kh,
        device=device,
        dtype=dtype,
        batch_size_pairs=batch_size_pairs,
        verbose=verbose,
    )
    return build_laplacian_from_distance(D_bi)


