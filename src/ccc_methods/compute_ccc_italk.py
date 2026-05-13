# ============================================================
# compute_ccc_italk.py
# Single-cell CCC using iTALK scoring:
#   P[i,j] = log(1 + L_i) + log(1 + R_j)
#
# Fully compatible with cccVAE BI-distance + Laplacian pipeline
# ============================================================

import os
import torch
import numpy as np
import pandas as pd


# ============================================================
# Device helpers
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
# Load iTALK LR DB
# ============================================================

def _load_italk_lr_db():
    """Load iTalk_LR_database.csv from same folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "iTalk_LR_database.csv")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Cannot find iTalk_LR_database.csv at:\n{path}\n"
            "Please place it in the same folder as compute_ccc_italk.py"
        )

    df = pd.read_csv(path)

    # required columns
    if not {"Ligand.ApprovedSymbol", "Receptor.ApprovedSymbol"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns: 'Ligand.ApprovedSymbol' and 'Receptor.ApprovedSymbol'.\n"
            f"Found columns: {df.columns.tolist()}"
        )

    return df


# ============================================================
# Prepare LR list
# ============================================================

def _prepare_LR_items_italk(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    device=None,
    dtype=None,
    verbose=True,
):
    """
    Build list of LR_items where each item contains:
      L_vals: (n_cells,)
      R_vals: (n_cells,)
      score : log1p(mean(L)) + log1p(mean(R))
    """
    device = _select_device(device)
    dtype  = _select_dtype(device, dtype)

    # convert expression matrix
    expr_t = torch.tensor(expr, device=device, dtype=dtype) \
             if isinstance(expr, np.ndarray) \
             else expr.to(device=device, dtype=dtype)

    n_cells, n_genes = expr_t.shape

    # build gene → vector dict
    gene_expr = {g: expr_t[:, i] for i, g in enumerate(genes)}

    df = _load_italk_lr_db()

    LR_items = []

    for _, row in df.iterrows():
        Lg = row["Ligand.ApprovedSymbol"]
        Rg = row["Receptor.ApprovedSymbol"]

        if Lg not in gene_expr or Rg not in gene_expr:
            continue  # skip missing genes

        L_vals = gene_expr[Lg]
        R_vals = gene_expr[Rg]

        # iTALK score
        score = np.log1p(L_vals.mean().item()) + np.log1p(R_vals.mean().item())

        if score < cutoff:
            continue

        LR_items.append(
            {
                "L_vals": L_vals,
                "R_vals": R_vals,
                "score":  score,
                "name":   f"{Lg}-{Rg}"
            }
        )

    if len(LR_items) == 0:
        raise ValueError(f"No valid LR pairs after cutoff={cutoff}")

    # sort by score (descending)
    LR_items = sorted(LR_items, key=lambda x: x["score"], reverse=True)

    if top_k is not None and top_k < len(LR_items):
        LR_items = LR_items[:top_k]

    return LR_items, n_cells, device, dtype


# ============================================================
# iTALK CCC → BI Frobenius Distance
# ============================================================

def compute_ccc_bi_distance_from_expr(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    alpha=0.5,
    Kh=1.0,       # unused, kept for API compatibility
    device="cuda",
    dtype=None,
    batch_size_pairs=4,
    verbose=True,
):
    LR_items, n_cells, device_t, dtype_t = _prepare_LR_items_italk(
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
        batch = LR_items[start: start + batch_size_pairs]

        L_batch = torch.stack([b["L_vals"] for b in batch]).to(device_t, dtype_t)
        R_batch = torch.stack([b["R_vals"] for b in batch]).to(device_t, dtype_t)

        # iTALK single-cell P[i,j] = log1p(L_i) + log1p(R_j)
        Li = torch.log1p(L_batch)[:, :, None]  # (B, n, 1)
        Rj = torch.log1p(R_batch)[:, None, :]  # (B, 1, n)
        P  = Li + Rj                           # (B, n, n)

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

        del L_batch, R_batch, Li, Rj, P, X_out, Gram_out, diag_out, contrib_out
        del X_in, Gram_in, diag_in, contrib_in

        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    eps = 1e-12
    D_out2 = torch.clamp(D_out2, min=0)
    D_in2  = torch.clamp(D_in2, min=0)

    D_bi2 = alpha * D_out2 + (1 - alpha) * D_in2
    D_bi  = torch.sqrt(D_bi2 + eps)

    D_bi_np = D_bi.detach().cpu().numpy().astype(np.float64)
    D_bi_np = 0.5 * (D_bi_np + D_bi_np.T)
    np.fill_diagonal(D_bi_np, 0)

    return D_bi_np


# ============================================================
# Build Laplacian
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


