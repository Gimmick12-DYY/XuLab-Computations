"""Unified gated TF imputation model — single-architecture replacement for the
scBasset_cisTopic_PUscOpen cascade.

Submodules:
    SequenceEncoder       Basset-style conv tower : (B, L, 4) one-hot -> (B, D)
    BinContext            Multi-scale dilated conv along genome order :
                              (n_bins, D) -> (n_bins, D)
                              effective context per branch = (kernel-1)*dilation
                              defaults [3kb, 9kb, 33kb] for 1 kb bins
                              per-channel residual gain `alpha`
    GatedUnifiedModel     SeqEncoder + BinContext + cell bank U + gate MLP
                              forward returns yhat_rc = g_rc * p_rc

Sequence input convention: (batch, seq_length, 4) one-hot float. The encoder
transposes internally to (B, 4, L) before Conv1d.

This file has no I/O or training logic — just nn.Modules and a tiny dimension
check at the bottom that runs if you `python -m unified.scripts._model`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- sequence encoder -------------------------------------------------

class SequenceEncoder(nn.Module):
    """Basset-family conv tower from one-hot DNA to a D-dim per-bin embedding.

    Structure (with default args):
        Conv1D(4, 288, k=17, pad=same) -> BN -> GELU -> MaxPool(3)
        Conv1D(288 -> 288 -> 323 -> 363 -> 407, k=5) -> BN -> GELU -> MaxPool(2)
        Conv1D(407, 256, k=1) -> BN -> GELU
        Flatten -> Dropout -> Linear(D=64) -> BN -> GELU

    For seq_length=768, the spatial dim after stem MaxPool(3) is 256; after 4
    further MaxPool(2) layers it lands on 16. Flattened size = 16 * 256 = 4096.
    """

    def __init__(self,
                 seq_length: int = 768,
                 latent_dim: int = 64,
                 stem_filters: int = 288,
                 stem_kernel: int = 17,
                 stem_pool: int = 3,
                 tower_channels: tuple[int, ...] = (288, 323, 363, 407),
                 tower_kernel: int = 5,
                 head_filters: int = 256,
                 dropout: float = 0.20):
        super().__init__()
        self.seq_length = int(seq_length)

        layers: list[nn.Module] = [
            nn.Conv1d(4, stem_filters, kernel_size=stem_kernel,
                      padding=stem_kernel // 2),
            nn.BatchNorm1d(stem_filters),
            nn.GELU(),
            nn.MaxPool1d(stem_pool),
        ]

        prev_ch = stem_filters
        for ch in tower_channels:
            layers += [
                nn.Conv1d(prev_ch, ch, kernel_size=tower_kernel,
                          padding=tower_kernel // 2),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            prev_ch = ch

        layers += [
            nn.Conv1d(prev_ch, head_filters, kernel_size=1),
            nn.BatchNorm1d(head_filters),
            nn.GELU(),
        ]
        self.tower = nn.Sequential(*layers)

        # Compute the spatial dim flowing through the tower.
        spatial = seq_length // stem_pool
        for _ in tower_channels:
            spatial //= 2
        if spatial <= 0:
            raise ValueError(
                f"seq_length={seq_length} too small for the configured pool chain "
                f"(stem_pool={stem_pool}, n_tower={len(tower_channels)})"
            )
        self._spatial_after_tower = spatial
        flat_size = spatial * head_filters

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_size, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 4) one-hot float -> (B, 4, L) for Conv1d
        if x.dim() != 3 or x.shape[-1] != 4:
            raise ValueError(f"SequenceEncoder expects (B, L, 4); got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.tower(x)        # (B, head_filters, spatial)
        x = self.head(x)         # (B, D)
        return x


# ---------- bin-axis context refinement --------------------------------------

class BinContext(nn.Module):
    """Multi-scale dilated 1-D conv along genome-ordered bins.

    Receptive fields (1 kb bins):
        d=1  k=3  -> 3 bins  (~3 kb)   sharp / promoter
        d=4  k=3  -> 9 bins  (~9 kb)   enhancer-class
        d=16 k=3  -> 33 bins (~33 kb)  broad-domain

    Inputs:
        e: (n_bins, D) for one chromosome (or chunk), bins sorted by start.

    Output:
        e + alpha (.) ctx(e)  -- per-channel residual gain.
    """

    def __init__(self,
                 dim: int = 64,
                 dilations: tuple[int, ...] = (1, 4, 16),
                 kernel: int = 3,
                 alpha_init: float = 0.1):
        super().__init__()
        if any(k % 2 != 1 for k in (kernel,)):
            raise ValueError("kernel must be odd for symmetric padding")
        self.dim = int(dim)
        self.dilations = tuple(int(d) for d in dilations)
        self.kernel = int(kernel)

        self.branches = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=kernel,
                      dilation=d, padding=(kernel - 1) // 2 * d)
            for d in self.dilations
        ])
        self.mixer = nn.Conv1d(dim * len(self.dilations), dim, kernel_size=1)
        # Per-channel residual gain. Init small so the model starts close to
        # pure sequence (no context) and learns how much context to mix in.
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        # e: (n_bins, dim) -> (1, dim, n_bins) for 1-D conv along chrom order
        if e.dim() != 2 or e.shape[-1] != self.dim:
            raise ValueError(f"BinContext expects (n_bins, {self.dim}); got {tuple(e.shape)}")
        x = e.transpose(0, 1).unsqueeze(0)               # (1, dim, n_bins)
        branches = [branch(x) for branch in self.branches]
        h = torch.cat(branches, dim=1)                   # (1, dim*B, n_bins)
        h = F.gelu(h)
        h = self.mixer(h)                                # (1, dim, n_bins)
        h = h.squeeze(0).transpose(0, 1)                 # (n_bins, dim)
        return e + self.alpha.unsqueeze(0) * h


# ---------- gated unified model ----------------------------------------------

class GatedUnifiedModel(nn.Module):
    """Joint sequence + bin-context + cell-bank + gate model.

    Forward pattern is two-stage by design:
      1. `encode_bins(seq)` produces `e_tilde` for a chunk of bins. This
         contains the sequence encoder + bin-context. Run once per chunk.
      2. `forward_pairs(e_tilde, bin_idx, cell_idx)` is called for arbitrary
         (r, c) pairs sampled within the chunk. Returns yhat = g * p plus the
         g and p for diagnostics.

    This split avoids re-running the conv tower for every sampled (r, c)
    pair -- only the per-(bin, cell) head is recomputed per pair.
    """

    def __init__(self,
                 n_cells: int,
                 seq_length: int = 768,
                 latent_dim: int = 64,
                 stem_filters: int = 288,
                 stem_kernel: int = 17,
                 stem_pool: int = 3,
                 tower_channels: tuple[int, ...] = (288, 323, 363, 407),
                 tower_kernel: int = 5,
                 head_filters: int = 256,
                 dropout: float = 0.20,
                 context_dilations: tuple[int, ...] = (1, 4, 16),
                 context_kernel: int = 3,
                 alpha_init: float = 0.1,
                 gate_hidden: tuple[int, ...] = (128, 64, 32),
                 gate_use_concat: bool = True,
                 cell_bank_init_std: float = 0.01):
        super().__init__()
        self.n_cells = int(n_cells)
        self.latent_dim = int(latent_dim)
        self.gate_use_concat = bool(gate_use_concat)

        self.seq_encoder = SequenceEncoder(
            seq_length=seq_length,
            latent_dim=latent_dim,
            stem_filters=stem_filters,
            stem_kernel=stem_kernel,
            stem_pool=stem_pool,
            tower_channels=tuple(tower_channels),
            tower_kernel=tower_kernel,
            head_filters=head_filters,
            dropout=dropout,
        )
        self.bin_context = BinContext(
            dim=latent_dim,
            dilations=tuple(context_dilations),
            kernel=context_kernel,
            alpha_init=alpha_init,
        )

        # Cell bank: learnable per-cell embedding + bias.
        self.U = nn.Parameter(
            torch.randn(n_cells, latent_dim) * float(cell_bank_init_std)
        )
        self.b = nn.Parameter(torch.zeros(n_cells))

        # Gate MLP: concat[ẽ ; u] -> 1 (or bilinear pool if gate_use_concat=False).
        gate_in = latent_dim * 2 if self.gate_use_concat else latent_dim
        prev = gate_in
        gate_layers: list[nn.Module] = []
        for h in gate_hidden:
            gate_layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        gate_layers += [nn.Linear(prev, 1)]
        self.gate_mlp = nn.Sequential(*gate_layers)

    # ---- forward by chunk: encode all bins in one chrom slice --------------

    def encode_bins(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (n_bins, seq_length, 4) -> ẽ_r: (n_bins, D).

        seq must be a contiguous chromosome chunk in genome order; the
        BinContext conv mixes within this dimension.
        """
        e = self.seq_encoder(seq)
        e_tilde = self.bin_context(e)
        return e_tilde

    # ---- forward per sampled (r, c) pair -----------------------------------

    def forward_pairs(self,
                      e_tilde: torch.Tensor,
                      bin_idx: torch.Tensor,
                      cell_idx: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (yhat, p, g) each of shape (n_pairs,).

        Args:
            e_tilde:  (n_bins_in_chunk, D), output of encode_bins.
            bin_idx:  (n_pairs,) int64, indices into e_tilde.
            cell_idx: (n_pairs,) int64, indices into self.U.
        """
        e = e_tilde[bin_idx]                  # (P, D)
        u = self.U[cell_idx]                  # (P, D)
        b = self.b[cell_idx]                  # (P,)

        p_logits = (e * u).sum(dim=-1) + b
        p = torch.sigmoid(p_logits)

        if self.gate_use_concat:
            gate_in = torch.cat([e, u], dim=-1)
        else:
            gate_in = e * u
        g = torch.sigmoid(self.gate_mlp(gate_in).squeeze(-1))

        yhat = g * p
        return yhat, p, g

    # ---- full-cell inference for one bin chunk -----------------------------

    def predict_chunk(self,
                      e_tilde: torch.Tensor,
                      cell_idx: torch.Tensor | None = None
                      ) -> torch.Tensor:
        """Predict yhat for every (bin in chunk, cell in cell_idx).

        Returns dense (n_bins_in_chunk, len(cell_idx)) tensor. Used at impute
        time, never at training time.
        """
        if cell_idx is None:
            U = self.U
            b = self.b
        else:
            U = self.U[cell_idx]
            b = self.b[cell_idx]

        # Bilinear prediction (n_bins, n_cells)
        p_logits = e_tilde @ U.T + b.unsqueeze(0)
        p = torch.sigmoid(p_logits)

        # Gate: per (bin, cell) -> need to construct concat features for all pairs.
        # We do this in cell-axis chunks to bound memory.
        n_bins, D = e_tilde.shape
        n_cells = U.shape[0]
        g = torch.empty(n_bins, n_cells, device=e_tilde.device, dtype=p.dtype)
        # Cell-axis chunking to bound concat tensor size.
        cell_chunk = max(1, min(1024, n_cells))
        for j0 in range(0, n_cells, cell_chunk):
            j1 = min(n_cells, j0 + cell_chunk)
            u_slice = U[j0:j1]                                    # (cj, D)
            if self.gate_use_concat:
                gate_in = torch.cat([
                    e_tilde.unsqueeze(1).expand(n_bins, j1 - j0, D),
                    u_slice.unsqueeze(0).expand(n_bins, j1 - j0, D),
                ], dim=-1)                                         # (n_bins, cj, 2D)
            else:
                gate_in = (e_tilde.unsqueeze(1) * u_slice.unsqueeze(0))
            g[:, j0:j1] = torch.sigmoid(self.gate_mlp(gate_in).squeeze(-1))
        return g * p

    # ---- L1 on alpha (encourages picking few scales per channel) ------------

    def alpha_l1(self) -> torch.Tensor:
        return self.bin_context.alpha.abs().sum()


# ---------- focal loss with sampled negatives --------------------------------

def focal_bce(yhat: torch.Tensor, y: torch.Tensor,
              gamma: float = 2.0, alpha: float = 0.25,
              eps: float = 1e-7) -> torch.Tensor:
    """Standard sigmoid focal loss.

    yhat, y in (0, 1); shapes match. Returns mean over the batch.
    Equivalent formulation to RetinaNet (Lin et al. 2017).
    """
    yhat = yhat.clamp(eps, 1.0 - eps)
    p_t = torch.where(y > 0.5, yhat, 1.0 - yhat)
    a_t = torch.where(y > 0.5, torch.full_like(yhat, alpha),
                      torch.full_like(yhat, 1.0 - alpha))
    loss = -a_t * (1.0 - p_t) ** gamma * torch.log(p_t)
    return loss.mean()


# ---------- self-check -------------------------------------------------------

def _smoke_check() -> None:
    """Tiny dimension check. Doesn't replace tests but catches shape bugs early."""
    torch.manual_seed(0)
    model = GatedUnifiedModel(n_cells=128, seq_length=768, latent_dim=64)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params/1e6:.2f}M")

    seq = torch.zeros(256, 768, 4)
    # plant random one-hots
    for i in range(256):
        idx = torch.randint(0, 4, (768,))
        seq[i].scatter_(1, idx.unsqueeze(-1), 1.0)

    e_tilde = model.encode_bins(seq)
    assert e_tilde.shape == (256, 64), f"e_tilde shape {tuple(e_tilde.shape)}"
    print(f"  e_tilde shape:    {tuple(e_tilde.shape)}")

    n_pairs = 1000
    bin_idx = torch.randint(0, 256, (n_pairs,))
    cell_idx = torch.randint(0, 128, (n_pairs,))
    yhat, p, g = model.forward_pairs(e_tilde, bin_idx, cell_idx)
    assert yhat.shape == (n_pairs,)
    print(f"  yhat range:       [{yhat.min().item():.4f}, {yhat.max().item():.4f}]")
    print(f"  p range:          [{p.min().item():.4f}, {p.max().item():.4f}]")
    print(f"  g range:          [{g.min().item():.4f}, {g.max().item():.4f}]")

    y = torch.bernoulli(torch.full((n_pairs,), 0.05))
    loss = focal_bce(yhat, y)
    loss.backward()
    print(f"  focal loss:       {loss.item():.4f}")
    print(f"  alpha (first 4):  {model.bin_context.alpha[:4].tolist()}")


if __name__ == "__main__":
    print("[unified._model] smoke check ...")
    _smoke_check()
    print("[unified._model] ok")
