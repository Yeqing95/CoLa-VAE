# ============================================================
# Cola_VAE.py main program
# ============================================================

import math
import os
from collections import deque

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.distributions import Normal, kl_divergence
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from I_PID import PIDControl
from VAE_utils import DenseEncoder, buildNetwork, MeanAct, NBLoss

# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
            return

        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss

# ============================================================
# COLAVAE
# ============================================================

class COLAVAE(nn.Module):
    def __init__(
        self,
        adata,
        input_dim,
        CCC_dim,
        Normal_dim,
        encoder_layers,
        decoder_layers,
        noise,
        encoder_dropout,
        decoder_dropout,
        KL_loss,
        dynamicVAE,
        init_beta,
        min_beta,
        max_beta,
        dtype,
        device,

        # HVG mask: np.ndarray of indices, or None(=all genes)
        hvg_idx=None,

        # CCC / methods
        ccc_builder=None,
        ccc_cutoff=0.1,
        ccc_top_k=None,
        ccc_alpha=0.5,
        ccc_Kh=1.0,
        ccc_batch_size_pairs=4,
        ccc_blockwise=False,
        ccc_cell_block_size=1024,

        # warmup
        warmup_epochs=20,
        ccc_update_interval=5,

        # Graph Laplacian λ
        laplacian_lambda_start=0.1,
        laplacian_lambda_max=1.0,
        laplacian_lambda_step=0.1,
        laplacian_lambda_update_interval=5,
    ):
        super().__init__()

        torch.set_default_dtype(dtype)
        self.dtype = dtype
        self.device = device

        # Latent dims
        if CCC_dim < 0:
            raise ValueError("CCC_dim cannot be a negative number.")
        self.CCC_dim = CCC_dim
        self.Normal_dim = Normal_dim
        self.latent_dim = CCC_dim + Normal_dim

        if CCC_dim == 0:
            print("[INFO] CCC_dim = 0: The model has degenerated into a standard VAE (without enabling the CCC structure prior).")

        # Encoder: full gene as input_dim
        self.encoder = DenseEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_layers,
            output_dim=self.latent_dim,
            activation="elu",
            dropout=encoder_dropout,
        )

        # Decoder: full gene as output_dim
        self.decoder = buildNetwork(
            [self.latent_dim] + decoder_layers,
            activation="elu",
            dropout=decoder_dropout,
        )
        if len(decoder_layers) > 0:
            self.dec_mean = nn.Sequential(
                nn.Linear(decoder_layers[-1], input_dim),
                MeanAct()
            )
        else:
            self.dec_mean = nn.Sequential(
                nn.Linear(self.latent_dim, input_dim),
                MeanAct()
            )

        # NB loss over genes (HVG masked)
        self.dec_disp = nn.Parameter(torch.randn(input_dim))
        self.NB_loss = NBLoss().to(device)

        # Dynamic beta (Normal KL)
        self.PID = PIDControl(
            Kp=0.01,
            Ki=-0.005,
            init_beta=init_beta,
            min_beta=min_beta,
            max_beta=max_beta,
        )
        self.KL_loss = KL_loss
        self.dynamicVAE = dynamicVAE
        self.beta = init_beta

        # Graph setting
        self.ccc_builder = ccc_builder
        self.ccc_cutoff = ccc_cutoff
        self.ccc_top_k = ccc_top_k
        self.ccc_alpha = ccc_alpha
        self.ccc_Kh = ccc_Kh
        self.ccc_batch_size_pairs = ccc_batch_size_pairs
        self.ccc_blockwise = ccc_blockwise
        self.ccc_cell_block_size = ccc_cell_block_size

        self.warmup_epochs = warmup_epochs
        self.ccc_update_interval = ccc_update_interval

        self.laplacian_lambda_start = laplacian_lambda_start
        self.laplacian_lambda_max = laplacian_lambda_max
        self.laplacian_lambda_step = laplacian_lambda_step
        self.laplacian_lambda_update_interval = laplacian_lambda_update_interval

        self.ccc_L = None
        self.lambda_graph = 0.0

        # gene names: full gene set after preprocess
        self.genes = list(adata.var_names)

        # HVG indices (used to mask NB loss). If None -> all genes
        if hvg_idx is None:
            self.hvg_idx = None
        else:
            self.hvg_idx = torch.tensor(hvg_idx, dtype=torch.long, device=device)

        # Encoder noise robustness
        self.noise = noise

        self.to(device)

    # ------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        d = torch.load(path, map_location="cpu")
        model_dict = self.state_dict()
        d = {k: v for k, v in d.items() if k in model_dict}
        model_dict.update(d)
        self.load_state_dict(model_dict)

    # ------------------------------------------------------
    # Update Laplacian
    # ------------------------------------------------------
    def update_laplacian(self, L_np):
        """
        L_np: np.ndarray, shape (n_cells, n_cells)，normalized Laplacian
        """
        if L_np is None:
            self.ccc_L = None
            return
        L = torch.tensor(L_np, dtype=self.dtype, device=self.device)
        L = 0.5 * (L + L.T)
        #L = torch.clamp(L, min=0.0)
        self.ccc_L = L

    # ======================================================
    # forward（single batch）
    # ======================================================
    def forward(self, x, y, raw_y, size_factors, num_samples=1):
        """
        Args:
            x: batch 中的 cell index（long），shape (b,)
            y: normalized counts (b, G_full)
            raw_y: raw counts (b, G_full)
            size_factors: size factors (b,)
        Returns:
            loss, recon_loss, kl_norm, graph_loss
        """

        b = y.shape[0]

        # Encoder
        q_mu, q_var = self.encoder(y)   # (b, latent_dim)

        if self.CCC_dim > 0:
            ccc_mu = q_mu[:, :self.CCC_dim]
            ccc_var = q_var[:, :self.CCC_dim]
            norm_mu = q_mu[:, self.CCC_dim:]
            norm_var = q_var[:, self.CCC_dim:]
        else:
            ccc_mu = None
            ccc_var = None
            norm_mu = q_mu
            norm_var = q_var

        # KL on Normal latent only
        prior_norm = Normal(
            torch.zeros_like(norm_mu),
            torch.ones_like(norm_var),
        )
        post_norm = Normal(
            norm_mu,
            torch.sqrt(norm_var + 1e-8),
        )
        kl_norm = kl_divergence(post_norm, prior_norm).sum()

        # Reconstruction over full genes, NB loss only on HVG
        post_all = Normal(q_mu, torch.sqrt(q_var + 1e-8))

        recon_loss = 0.0
        for _ in range(num_samples):
            z = post_all.rsample()
            h = self.decoder(z)
            mean = self.dec_mean(h)                                # (b, G_full)
            disp = torch.exp(torch.clamp(self.dec_disp, -15., 15.)).unsqueeze(0)  # (1, G_full)

            if self.hvg_idx is not None:
                idx = self.hvg_idx
                x_nb = raw_y[:, idx]
                mean_nb = mean[:, idx]
                disp_nb = disp[:, idx]
            else:
                x_nb = raw_y
                mean_nb = mean
                disp_nb = disp

            recon_loss += self.NB_loss(
                x=x_nb,
                mean=mean_nb,
                disp=disp_nb,
                scale_factor=size_factors,
            )
        recon_loss /= num_samples

        # Noise robustness
        noise_reg = 0.0
        if self.noise > 0:
            for _ in range(num_samples):
                y_noisy = y + torch.randn_like(y) * self.noise
                q_mu_n, _ = self.encoder(y_noisy)
                noise_reg += torch.sum((q_mu - q_mu_n) ** 2)
            noise_reg /= num_samples

        # Graph Laplacian prior on z_CCC (posterior mean)
        if self.CCC_dim > 0 and self.ccc_L is not None and self.lambda_graph > 0:
            idx_cells = x.long()
            L_bb = self.ccc_L.index_select(0, idx_cells).index_select(1, idx_cells)
            Z_c = ccc_mu
            graph_loss = torch.trace(Z_c.t() @ L_bb @ Z_c) / float(b)
        else:
            graph_loss = 0.0

        # Total loss
        loss = recon_loss + self.beta * kl_norm

        if self.noise > 0:
            loss = loss + noise_reg * y.shape[1] / float(self.latent_dim)

        if self.CCC_dim > 0 and self.ccc_L is not None and self.lambda_graph > 0:
            loss = loss + self.lambda_graph * graph_loss

        return loss, recon_loss, kl_norm, graph_loss

    # ======================================================
    # Batch tool：latent & denoise
    # ======================================================

    def batching_latent_samples(self, X, Y, num_samples=1, batch_size=512):
        self.eval()
        Y = torch.tensor(Y, dtype=self.dtype)
        latents = []
        for i in range(0, Y.shape[0], batch_size):
            yb = Y[i:i+batch_size].to(self.device)
            q_mu, q_var = self.encoder(yb)
            dist = Normal(q_mu, torch.sqrt(q_var + 1e-8))
            for _ in range(num_samples):
                z = dist.rsample()
                latents.append(z.detach().cpu())
        return torch.cat(latents, dim=0).numpy()

    def batching_denoise_counts(self, X, Y, n_samples=5, batch_size=512):
        self.eval()
        Y = torch.tensor(Y, dtype=self.dtype)
        out = []
        for i in range(0, Y.shape[0], batch_size):
            yb = Y[i:i+batch_size].to(self.device)
            q_mu, q_var = self.encoder(yb)
            dist = Normal(q_mu, torch.sqrt(q_var + 1e-8))
            means = []
            for _ in range(n_samples):
                z = dist.rsample()
                h = self.decoder(z)
                m = self.dec_mean(h)
                means.append(m)
            means = torch.stack(means, dim=0).mean(0)
            out.append(means.detach().cpu())
        return torch.cat(out, dim=0).numpy()

    # ======================================================
    # Training
    # ======================================================

    def train_model(
        self,
        pos,
        ncounts,
        raw_counts,
        size_factors,
        lr=0.001,
        weight_decay=0.001,
        batch_size=512,
        num_samples=1,
        train_size=0.95,
        maxiter=500,
        patience=50,
        save_model=True,
        model_weights="colavae.pt",
        save_latent_interval=None,
        latent_prefix="latent",
        output_prefix="colavae",
        use_scheduler=False,
        scheduler_type="cosine",
        scheduler_step=10,
        scheduler_gamma=0.5,
        seed=2026,
    ):

        if sp.issparse(ncounts):
            ncounts = ncounts.toarray()
        if sp.issparse(raw_counts):
            raw_counts = raw_counts.toarray()

        pos = torch.tensor(pos, dtype=torch.int64)

        dataset = TensorDataset(
            pos,
            torch.tensor(ncounts, dtype=self.dtype),
            torch.tensor(raw_counts, dtype=self.dtype),
            torch.tensor(size_factors, dtype=self.dtype),
        )

        if train_size < 1:
            n_total = len(dataset)
            n_train = int(n_total * train_size)
            n_val = n_total - n_train
            train_ds, valid_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
            valid_loader = DataLoader(valid_ds, batch_size=batch_size)
        else:
            train_ds = dataset
            valid_loader = None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,generator=torch.Generator().manual_seed(seed))
        early_stop = EarlyStopping(patience=patience, modelfile=model_weights)

        opt = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # scheduler
        scheduler = None
        if use_scheduler:
            if scheduler_type == "cosine":
                scheduler = CosineAnnealingLR(opt, T_max=maxiter)
            elif scheduler_type == "step":
                scheduler = StepLR(opt, step_size=scheduler_step, gamma=scheduler_gamma)
            elif scheduler_type == "plateau":
                scheduler = ReduceLROnPlateau(opt, factor=scheduler_gamma, patience=scheduler_step)
            print(f"[Scheduler] Enabled: {scheduler_type}")

        # log and latent
        log_dir = "logs"
        latent_dir = os.path.join("latent", output_prefix)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)
        loss_log_path = os.path.join(log_dir, f"{output_prefix}_loss_log.txt")
        loss_log_f = open(loss_log_path, "w")
        loss_log_f.write("epoch\tELBO\trecon\tKL\tgraph\tbeta\tlambda_graph\n")

        queue = deque(maxlen=10)
        n_cells = ncounts.shape[0]

        print("Training...")

        for epoch in range(maxiter):
            self.train()
            total_loss = 0.0
            total_recon = 0.0
            total_kl = 0.0
            total_graph = 0.0
            N_samples = 0

            for xb, yb, yrb, sf in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                yrb = yrb.to(self.device)
                sf = sf.to(self.device)

                loss, rec, kl_norm, graph_loss = self.forward(
                    x=xb,
                    y=yb,
                    raw_y=yrb,
                    size_factors=sf,
                    num_samples=num_samples,
                )

                self.zero_grad()
                loss.backward()
                opt.step()

                bs = xb.shape[0]
                total_loss += loss.item()
                total_recon += rec.item()
                total_kl += kl_norm.item()
                
                try:
                    total_graph += graph_loss.detach().item()
                except AttributeError:
                    total_graph += float(graph_loss)

                N_samples += bs

                if self.dynamicVAE:
                    KL_val = kl_norm.item() / bs
                    queue.append(KL_val)
                    avg_KL = np.mean(queue)
                    target_KL = self.KL_loss * self.Normal_dim
                    self.beta, _ = self.PID.pid(target_KL, avg_KL)

            # epoch avg
            avg_loss = total_loss / N_samples
            avg_recon = total_recon / N_samples
            avg_kl = total_kl / N_samples
            avg_graph = total_graph / N_samples

            epoch_ELBO = -avg_loss

            print(
                f"Epoch {epoch+1}: "
                f"ELBO={epoch_ELBO:.4f}, "
                f"Recon={avg_recon:.4f}, "
                f"KL={avg_kl:.4f}, "
                f"Graph={avg_graph:.4f}, "
                f"beta={self.beta:.4f}, "
                f"lambda={self.lambda_graph:.4f}"
            )
            
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
                print(f"    → LR: {current_lr:.6f}")
            
            loss_log_f.write(
                f"{epoch+1}\t{epoch_ELBO}\t{avg_recon}\t{avg_kl}\t{avg_graph}\t{self.beta}\t{self.lambda_graph}\n"
            )
            loss_log_f.flush()

            # =====================================================
            # Periodically update CCC Laplacian after warm-up phase
            # =====================================================
            if self.CCC_dim > 0 and self.ccc_builder is not None:
                if (
                    (epoch + 1) >= self.warmup_epochs
                    and ((epoch + 1 - self.warmup_epochs) % self.ccc_update_interval == 0)
                ):
                    print(f"[CCC] Updating graph Laplacian at epoch {epoch+1} ...")

                    X_index = np.arange(n_cells, dtype=int)
                    denoised = self.batching_denoise_counts(
                        X=X_index,
                        Y=ncounts,
                        n_samples=5,
                        batch_size=batch_size,
                    )

                    L_new = self.ccc_builder(
                        expr=denoised,
                        genes=self.genes,
                        cutoff=self.ccc_cutoff,
                        top_k=self.ccc_top_k,
                        alpha=self.ccc_alpha,
                        Kh=self.ccc_Kh,
                        device=str(self.device),
                        dtype="float64",
                        batch_size_pairs=self.ccc_batch_size_pairs,
                        blockwise=self.ccc_blockwise,
                        cell_block_size=self.ccc_cell_block_size,
                        verbose=True,
                    )
                    self.update_laplacian(L_new)

                    # λ scheduling
                    if self.lambda_graph == 0.0:
                        self.lambda_graph = self.laplacian_lambda_start
                    else:
                        if ((epoch + 1 - self.warmup_epochs) % self.laplacian_lambda_update_interval) == 0:
                            self.lambda_graph = min(
                                self.lambda_graph + self.laplacian_lambda_step,
                                self.laplacian_lambda_max,
                            )

            # Validation
            if valid_loader is not None:
                self.eval()
                val_loss = 0.0
                val_N = 0
                with torch.no_grad():
                    for xb, yb, yrb, sf in valid_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        yrb = yrb.to(self.device)
                        sf = sf.to(self.device)

                        loss, _, _, _ = self.forward(
                            x=xb,
                            y=yb,
                            raw_y=yrb,
                            size_factors=sf,
                            num_samples=num_samples,
                        )
                        val_loss += loss.item()
                        val_N += xb.shape[0]
                val_loss /= val_N
                val_ELBO = -val_loss
                print(f"Validation ELBO: {val_ELBO:.4f}")

                early_stop(val_loss, self)
                if early_stop.early_stop:
                    print("Early stopping!")
                    break

                # scheduler for plateau
                if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)

            # scheduler for non-plateau
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            # save latent（option）
            if (save_latent_interval is not None) and (save_latent_interval > 0) and ((epoch + 1) % save_latent_interval == 0):
                latent_np = self.batching_latent_samples(
                    X=pos.numpy(),
                    Y=ncounts,
                    num_samples=1,
                    batch_size=batch_size,
                )
                latent_path = os.path.join(
                    latent_dir,
                    f"{latent_prefix}_epoch_{epoch+1}.txt"
                )
                np.savetxt(latent_path, latent_np, delimiter=",")

        loss_log_f.close()

        if save_model:
            torch.save(self.state_dict(), model_weights)


