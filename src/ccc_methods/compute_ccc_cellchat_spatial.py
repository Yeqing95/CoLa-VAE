# ============================================================
# compute_ccc_cellchat_spatial.py
# ============================================================

import os
import re
import torch
import numpy as np
import pandas as pd


# ============================================================
# Utility parsing
# ============================================================

def _parse_genes(s):
    if pd.isna(s) or s == "":
        return []
    parts = re.split(r"[;,|+]+", str(s))
    return [x.strip() for x in parts if x.strip()]


def _normalize_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _normalize_dtype(dtype):
    if dtype is None:
        return torch.float64
    if isinstance(dtype, torch.dtype):
        return dtype
    s = str(dtype).lower()
    if s in ("float32", "float"):
        return torch.float32
    if s in ("float64", "double"):
        return torch.float64
    raise ValueError(f"Unsupported dtype string: {dtype}")


def _geometric_mean(vectors):
    X = torch.stack(vectors, dim=0)      # (k, n_cells)
    return torch.exp(torch.mean(torch.log(X + 1e-9), dim=0))


def _geo_mean_for_list(
    gene_list,
    gene_expr,
    n_cells,
    device,
    dtype,
    allow_missing=False,
):
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

    return _geometric_mean(vecs)


# ============================================================
# 1. CellChat
# ============================================================

def _default_lrdb_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "CellChatDB.mouse.v1.csv")
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"Default CellChatDB.human.v1.csv not found at: {candidate}\n"
            f"Please put CellChatDB.human.v1.csv in the same folder as compute_ccc_cellchat_spatial.py."
        )
    return candidate


def _prepare_LR_items_cellchat(
    expr,
    genes,
    lrdb_path=None,
    top_k=None,
    cutoff=0.1,
    device='cuda',
    dtype=torch.float64,
    Kh=1.0,
    verbose=True,
):
    device = _normalize_device(device)
    dtype = _normalize_dtype(dtype)

    # expr -> torch
    if isinstance(expr, np.ndarray):
        expr_t = torch.tensor(expr, device=device, dtype=dtype)
    else:
        expr_t = expr.to(device=device, dtype=dtype)

    n_cells, _ = expr_t.shape
    gene_expr = {g: expr_t[:, i] for i, g in enumerate(genes)}

    # load LR DB
    if lrdb_path is None:
        lrdb_path = _default_lrdb_path()
    df = pd.read_csv(lrdb_path)
    if verbose:
        print(f"Loaded CellChat LR database from {lrdb_path}, total {len(df)} interactions.")

    LR_items = []

    for _, row in df.iterrows():
        L_genes  = _parse_genes(row.get("ligand", ""))
        R_genes  = _parse_genes(row.get("receptor", ""))
        AG_genes = _parse_genes(row.get("agonist", ""))
        AN_genes = _parse_genes(row.get("antagonist", ""))
        RA_genes = _parse_genes(row.get("co_A_receptor", ""))
        RI_genes = _parse_genes(row.get("co_I_receptor", ""))

        L_vals = _geo_mean_for_list(L_genes, gene_expr, n_cells, device, dtype, allow_missing=False)
        if L_vals is None:
            continue

        R_core = _geo_mean_for_list(R_genes, gene_expr, n_cells, device, dtype, allow_missing=False)
        if R_core is None:
            continue

        # co-stimulatory / co-inhibitory receptor
        RA_vals = _geo_mean_for_list(RA_genes, gene_expr, n_cells, device, dtype, allow_missing=True)
        RI_vals = _geo_mean_for_list(RI_genes, gene_expr, n_cells, device, dtype, allow_missing=True)
        R_vals = R_core * (1.0 + RA_vals) / (1.0 + RI_vals)

        # agonist / antagonist
        AG_vals = _geo_mean_for_list(AG_genes, gene_expr, n_cells, device, dtype, allow_missing=True)
        AN_vals = _geo_mean_for_list(AN_genes, gene_expr, n_cells, device, dtype, allow_missing=True)

        L_mean = L_vals.mean().item()
        R_mean = R_vals.mean().item()
        score = min(L_mean, R_mean)

        if score < cutoff:
            continue

        LR_items.append(
            {
                "L_vals":  L_vals,
                "R_vals":  R_vals,
                "AG_vals": AG_vals,
                "AN_vals": AN_vals,
                "score":   score,
                "name":    row.get("interaction_name", ""),
            }
        )

    if len(LR_items) == 0:
        raise ValueError(
            f"No valid LR pairs after cutoff={cutoff}. "
            f"Check LR DB and gene names."
        )

    LR_items = sorted(LR_items, key=lambda x: x["score"], reverse=True)

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive or None.")
        if top_k < len(LR_items):
            LR_items = LR_items[:top_k]

    return LR_items, n_cells, device, dtype


# ============================================================
# 2. Strict CellChat formula -> BI Frobenius distance
# ============================================================

def compute_ccc_bi_distance_from_expr(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    alpha=0.5,
    Kh=1.0,
    device=None,
    dtype=None,
    batch_size_pairs=4,
    verbose=False,
):
    """
    Standard implementation.
    More memory-friendly than the original by keeping only one (B, n, n)
    interaction tensor P at a time, but still scales poorly for very large n.
    """
    LR_items, n_cells, device_t, dtype_t = _prepare_LR_items_cellchat(
        expr=expr,
        genes=genes,
        lrdb_path=None,
        top_k=top_k,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        Kh=Kh,
        verbose=verbose,
    )

    D_out2 = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)
    D_in2  = torch.zeros((n_cells, n_cells), device=device_t, dtype=dtype_t)

    K = len(LR_items)

    for start in range(0, K, batch_size_pairs):
        end = min(start + batch_size_pairs, K)
        batch = LR_items[start:end]
        B = len(batch)
        if B == 0:
            continue

        L_batch  = torch.stack([item["L_vals"]  for item in batch], dim=0).to(device_t, dtype_t)
        R_batch  = torch.stack([item["R_vals"]  for item in batch], dim=0).to(device_t, dtype_t)
        AG_batch = torch.stack([item["AG_vals"] for item in batch], dim=0).to(device_t, dtype_t)
        AN_batch = torch.stack([item["AN_vals"] for item in batch], dim=0).to(device_t, dtype_t)

        Li = L_batch[:, :, None]        # (B, n, 1)
        Rj = R_batch[:, None, :]        # (B, 1, n)
        AG_i = AG_batch[:, :, None]
        AG_j = AG_batch[:, None, :]
        AN_i = AN_batch[:, :, None]
        AN_j = AN_batch[:, None, :]

        # memory-friendlier version: keep only one (B,n,n) tensor P
        LR = Li * Rj
        P = LR / (Kh + LR)
        del LR

        P *= (1.0 + AG_i / (Kh + AG_i))
        P *= (1.0 + AG_j / (Kh + AG_j))
        P *= (Kh / (Kh + AN_i))
        P *= (Kh / (Kh + AN_j))

        # OUT profile
        X_out = P.reshape(B * n_cells, n_cells)
        Gram_out = X_out.T @ X_out
        diag_out = torch.diag(Gram_out)
        contrib_out = diag_out[:, None] + diag_out[None, :] - 2.0 * Gram_out
        D_out2 += contrib_out

        # IN profile
        X_in = P.permute(1, 0, 2).reshape(n_cells, B * n_cells)
        Gram_in = X_in @ X_in.T
        diag_in = torch.diag(Gram_in)
        contrib_in = diag_in[:, None] + diag_in[None, :] - 2.0 * Gram_in
        D_in2 += contrib_in

        del L_batch, R_batch, AG_batch, AN_batch
        del Li, Rj, AG_i, AG_j, AN_i, AN_j, P
        del X_out, Gram_out, diag_out, contrib_out, X_in, Gram_in, diag_in, contrib_in
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    eps = 1e-12
    D_out2 = torch.clamp(D_out2, min=0.0)
    D_in2  = torch.clamp(D_in2,  min=0.0)

    D_bi2 = alpha * D_out2 + (1.0 - alpha) * D_in2
    D_bi2 = torch.clamp(D_bi2, min=0.0)
    D_bi  = torch.sqrt(D_bi2 + eps)

    D_bi_np = D_bi.detach().cpu().numpy().astype(np.float64)
    D_bi_np = 0.5 * (D_bi_np + D_bi_np.T)
    np.fill_diagonal(D_bi_np, 0.0)

    D_bi_np[D_bi_np < 0] = 0.0
    finite_mask = np.isfinite(D_bi_np)
    if not finite_mask.all():
        finite_max = np.max(D_bi_np[finite_mask])
        D_bi_np[~finite_mask] = finite_max

    return D_bi_np


@torch.no_grad()
def compute_ccc_bi_distance_from_expr_blockwise(
    expr,
    genes,
    top_k=None,
    cutoff=0.1,
    alpha=0.5,
    Kh=1.0,
    device=None,
    dtype=None,
    batch_size_pairs=4,
    cell_block_size=1024,
    verbose=False,
):
    """
    Block-wise implementation for large datasets.

    Key idea:
      - never materialize a full (B, n_cells, n_cells) tensor P
      - for OUT profile, process sender cells in blocks
      - for IN profile, process receiver cells in blocks

    This preserves the same math as the standard implementation
    (up to floating-point roundoff), while substantially reducing peak memory.
    """
    LR_items, n_cells, device_t, dtype_t = _prepare_LR_items_cellchat(
        expr=expr,
        genes=genes,
        lrdb_path=None,
        top_k=top_k,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        Kh=Kh,
        verbose=verbose,
    )

    acc_dtype = dtype_t
    D_out2 = torch.zeros((n_cells, n_cells), device=device_t, dtype=acc_dtype)
    D_in2  = torch.zeros((n_cells, n_cells), device=device_t, dtype=acc_dtype)

    K = len(LR_items)

    for start in range(0, K, batch_size_pairs):
        end = min(start + batch_size_pairs, K)
        batch = LR_items[start:end]
        B = len(batch)
        if B == 0:
            continue

        if verbose:
            print(f"  Processing LR batch {start}:{end} (B={B})")

        L_batch  = torch.stack([item["L_vals"]  for item in batch], dim=0).to(device_t, dtype_t)
        R_batch  = torch.stack([item["R_vals"]  for item in batch], dim=0).to(device_t, dtype_t)
        AG_batch = torch.stack([item["AG_vals"] for item in batch], dim=0).to(device_t, dtype_t)
        AN_batch = torch.stack([item["AN_vals"] for item in batch], dim=0).to(device_t, dtype_t)

        # 1) OUT profile: block sender cells
        for i_start in range(0, n_cells, cell_block_size):
            i_end = min(i_start + cell_block_size, n_cells)
            m = i_end - i_start

            L_i  = L_batch[:, i_start:i_end]
            AG_i = AG_batch[:, i_start:i_end]
            AN_i = AN_batch[:, i_start:i_end]

            Li = L_i[:, :, None]               # (B, m, 1)
            Rj = R_batch[:, None, :]           # (B, 1, n)

            P_block = Li * Rj                  # (B, m, n)
            P_block = P_block / (Kh + P_block)

            AG_i_block = AG_i[:, :, None]      # (B, m, 1)
            AG_j_block = AG_batch[:, None, :]  # (B, 1, n)
            P_block *= (1.0 + AG_i_block / (Kh + AG_i_block))
            P_block *= (1.0 + AG_j_block / (Kh + AG_j_block))

            AN_i_block = AN_i[:, :, None]      # (B, m, 1)
            AN_j_block = AN_batch[:, None, :]  # (B, 1, n)
            P_block *= (Kh / (Kh + AN_i_block))
            P_block *= (Kh / (Kh + AN_j_block))

            X_out_block = P_block.reshape(B * m, n_cells)
            Gram_out_block = X_out_block.T @ X_out_block
            D_out2 += Gram_out_block

            del L_i, AG_i, AN_i, Li, Rj
            del P_block, AG_i_block, AG_j_block, AN_i_block, AN_j_block
            del X_out_block, Gram_out_block
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

        # 2) IN profile: block receiver cells
        for j_start in range(0, n_cells, cell_block_size):
            j_end = min(j_start + cell_block_size, n_cells)
            m = j_end - j_start

            R_j  = R_batch[:, j_start:j_end]
            AG_j = AG_batch[:, j_start:j_end]
            AN_j = AN_batch[:, j_start:j_end]

            Li = L_batch[:, :, None]           # (B, n, 1)
            Rj = R_j[:, None, :]               # (B, 1, m)

            P_block = Li * Rj                  # (B, n, m)
            P_block = P_block / (Kh + P_block)

            AG_i_full = AG_batch[:, :, None]   # (B, n, 1)
            AG_j_block = AG_j[:, None, :]      # (B, 1, m)
            P_block *= (1.0 + AG_i_full / (Kh + AG_i_full))
            P_block *= (1.0 + AG_j_block / (Kh + AG_j_block))

            AN_i_full = AN_batch[:, :, None]   # (B, n, 1)
            AN_j_block = AN_j[:, None, :]      # (B, 1, m)
            P_block *= (Kh / (Kh + AN_i_full))
            P_block *= (Kh / (Kh + AN_j_block))

            X_in_block = P_block.permute(1, 0, 2).reshape(n_cells, B * m)
            Gram_in_block = X_in_block @ X_in_block.T
            D_in2 += Gram_in_block

            del R_j, AG_j, AN_j, Li, Rj
            del P_block, AG_i_full, AG_j_block, AN_i_full, AN_j_block
            del X_in_block, Gram_in_block
            if device_t.type == "cuda":
                torch.cuda.empty_cache()

        del L_batch, R_batch, AG_batch, AN_batch
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    # Convert accumulated Gram matrices into squared Euclidean distances
    eps = 1e-12
    D_out2 = torch.clamp(D_out2, min=0.0)
    D_in2  = torch.clamp(D_in2,  min=0.0)

    diag_out = torch.diag(D_out2)
    D_out2 = diag_out[:, None] + diag_out[None, :] - 2.0 * D_out2

    diag_in = torch.diag(D_in2)
    D_in2 = diag_in[:, None] + diag_in[None, :] - 2.0 * D_in2

    D_out2 = torch.clamp(D_out2, min=0.0)
    D_in2  = torch.clamp(D_in2,  min=0.0)

    D_bi2 = alpha * D_out2 + (1.0 - alpha) * D_in2
    D_bi2 = torch.clamp(D_bi2, min=0.0)
    D_bi  = torch.sqrt(D_bi2 + eps)

    D_bi_np = D_bi.detach().cpu().numpy().astype(np.float64)
    D_bi_np = 0.5 * (D_bi_np + D_bi_np.T)
    np.fill_diagonal(D_bi_np, 0.0)

    D_bi_np[D_bi_np < 0] = 0.0
    finite_mask = np.isfinite(D_bi_np)
    if not finite_mask.all():
        finite_max = np.max(D_bi_np[finite_mask])
        D_bi_np[~finite_mask] = finite_max

    return D_bi_np




# ============================================================
# Spatial mask utilities
# ============================================================

def compute_spatial_dist_matrix(coords):
    """
    Compute Euclidean spatial distance matrix from spatial coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (n_cells, 2) or (n_cells, 3)

    Returns
    -------
    D_spatial : np.ndarray, shape (n_cells, n_cells)
    """
    coords = np.asarray(coords, dtype=np.float64)
    sq = np.sum(coords ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (coords @ coords.T)
    D2[D2 < 0.0] = 0.0
    return np.sqrt(D2)


def build_spatial_mask(
    coords,
    radius_cutoff=100.0,
    spatial_scale=50.0,
    mode="exp",
):
    """
    Build a spatial mask M in [0, 1].

    The mask is applied to the CCC similarity matrix W/S before the
    graph Laplacian is computed:

        S_spatial = S_ccc * M_spatial
        L_spatial_ccc = Laplacian(S_spatial)

    Parameters
    ----------
    coords : np.ndarray, shape (n_cells, 2/3)
    radius_cutoff : float or None
        Maximum spatial distance to keep an edge. If None, use median
        non-zero pairwise distance.
    spatial_scale : float
        Scale for exponential decay.
    mode : str
        "exp", "linear", or "binary".

    Returns
    -------
    M : np.ndarray, shape (n_cells, n_cells)
    """
    D = compute_spatial_dist_matrix(coords)
    n = D.shape[0]

    if radius_cutoff is None:
        off_diag = D[np.triu_indices(n, k=1)]
        off_diag = off_diag[off_diag > 0]
        radius_cutoff = float(np.median(off_diag)) if off_diag.size > 0 else 1.0

    inside = D <= float(radius_cutoff)

    if mode == "exp":
        if spatial_scale is None or spatial_scale <= 0:
            spatial_scale = float(radius_cutoff)
        M = np.exp(-D / float(spatial_scale))
        M[~inside] = 0.0
    elif mode == "linear":
        M = 1.0 - D / float(radius_cutoff)
        M[M < 0.0] = 0.0
    elif mode == "binary":
        M = inside.astype(np.float64)
    else:
        raise ValueError(f"Unknown spatial_mask_mode: {mode}. Choose from exp, linear, binary.")

    M = 0.5 * (M + M.T)
    np.fill_diagonal(M, 0.0)
    return M.astype(np.float64)


# ============================================================
# 3. From D_bi to spatial-masked symmetric normalized Laplacian
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

def build_laplacian_from_distance(
    D_bi,
    spatial_coords=None,
    spatial_radius_cutoff=100.0,
    spatial_scale=50.0,
    spatial_mask_mode="exp",
    topk_sparsify=30,
    verbose=False,
):
    """
    Convert CellChat BI distance to a symmetric normalized Laplacian.

    If spatial_coords is provided, the CCC similarity matrix is spatially
    pruned/weighted BEFORE computing the Laplacian. This is the key spatial
    version:

        D_bi -> S_ccc -> S_ccc * M_spatial -> L

    Do not directly mask the final Laplacian.
    """
    D = np.asarray(D_bi, dtype=np.float64)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0

    finite_mask = np.isfinite(D)
    if not finite_mask.all():
        finite_vals = D[finite_mask]
        finite_max = np.max(finite_vals) if finite_vals.size > 0 else 1.0
        D[~finite_mask] = finite_max

    pos = D[D > 0]
    if pos.size == 0:
        n = D.shape[0]
        return np.zeros((n, n), dtype=np.float64)

    sigma = np.median(pos)
    if sigma <= 0:
        sigma = np.mean(pos) if np.mean(pos) > 0 else 1.0

    # CCC similarity from BI distance
    S = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
    S = 0.5 * (S + S.T)
    S[S < 0] = 0.0
    np.fill_diagonal(S, 0.0)

    # Spatial masking / pruning on similarity graph, not on Laplacian
    if spatial_coords is not None:
        M = build_spatial_mask(
            coords=spatial_coords,
            radius_cutoff=spatial_radius_cutoff,
            spatial_scale=spatial_scale,
            mode=spatial_mask_mode,
        )
        if M.shape != S.shape:
            raise ValueError(
                f"Spatial mask shape {M.shape} does not match CCC similarity shape {S.shape}."
            )
        S = S * M
        S = 0.5 * (S + S.T)
        np.fill_diagonal(S, 0.0)

        if verbose:
            nnz = np.count_nonzero(S)
            total = S.size - S.shape[0]
            density = nnz / max(total, 1)
            print(
                f"[Spatial mask] mode={spatial_mask_mode}, "
                f"radius={spatial_radius_cutoff}, scale={spatial_scale}, "
                f"CCC graph density after mask={density:.6f}"
            )

    if topk_sparsify is not None and topk_sparsify > 0:
        S = sparsify_topk(S, k=topk_sparsify)

    deg = S.sum(axis=1)

    # If spatial pruning creates isolated nodes, keep them as isolated.
    # The normalized Laplacian diagonal remains 1 for isolated nodes.
    safe_deg = deg.copy()
    safe_deg[safe_deg <= 1e-12] = 1e-12
    D_inv_sqrt = 1.0 / np.sqrt(safe_deg)

    n = S.shape[0]
    I = np.eye(n, dtype=np.float64)
    D_inv_sqrt_mat = np.diag(D_inv_sqrt)
    L = I - D_inv_sqrt_mat @ S @ D_inv_sqrt_mat
    L = 0.5 * (L + L.T)

    return L


# ============================================================
# 4. Public interface: build Laplacian for cccVAE
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
    spatial_coords=None,
    spatial_radius_cutoff=100.0,
    spatial_scale=50.0,
    spatial_mask_mode="exp",
    topk_sparsify=30,
    verbose=False,
):
    if blockwise:
        D_bi = compute_ccc_bi_distance_from_expr_blockwise(
            expr=expr,
            genes=genes,
            top_k=top_k,
            cutoff=cutoff,
            alpha=alpha,
            Kh=Kh,
            device=device,
            dtype=dtype,
            batch_size_pairs=batch_size_pairs,
            cell_block_size=cell_block_size,
            verbose=verbose,
        )
    else:
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

    L_np = build_laplacian_from_distance(
        D_bi,
        spatial_coords=spatial_coords,
        spatial_radius_cutoff=spatial_radius_cutoff,
        spatial_scale=spatial_scale,
        spatial_mask_mode=spatial_mask_mode,
        topk_sparsify=topk_sparsify,
        verbose=verbose,
    )
    return L_np

