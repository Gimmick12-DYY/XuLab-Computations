#!/usr/bin/env python
# -----------------------------------------------------------------------------
# 03_train.py
#
# Self-contained Keras implementation of scBasset (Yuan & Kelley, Nat. Methods
# 2022). The model maps a 1344-bp one-hot sequence to a 32-d embedding via a
# Basset-style conv tower, then projects it through a (latent_dim, n_cells)
# dense head with a sigmoid output. Cell embeddings are the dense head's
# weight matrix W (shape (latent_dim, n_cells)) -- they are the byproduct of
# training, not a separate model component.
#
# Inputs (from 02_prepare_seqs.py):
#   <work>/seqs/seqs.h5            X, trainable, ...
#   <work>/seqs/matrix_train.npz   sparse CSR (n_trainable_bins, n_cells), binary
#   <work>/seqs/barcodes.tsv       cell order
#   <work>/seqs/trainable_idx_in_h5.npy  rows of seqs.h5/X used for training
#
# Outputs (under <work>/model/):
#   scbasset.keras            full Keras model (architecture + weights)
#   best.weights.h5           best weights by val loss
#   cell_embeddings.npz       W = (n_cells, latent_dim), b = (n_cells,)
#   history.json              per-epoch losses
#   train_meta.json           hyperparameters + sizes
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from _cfg import base_parser, load_config, resolve_paths

# TF is loaded lazily so non-GPU steps in this pipeline don't pull it in.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

log = logging.getLogger("03_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def build_scbasset(n_cells: int, seq_length: int, latent_dim: int,
                   tower_channels: list[int], dropout_rate: float,
                   l2: float):
    """Compact scBasset-style model -- conv tower -> seq embed -> cell-wise dense."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers, Input, Model

    reg = regularizers.l2(l2) if l2 > 0 else None

    inputs = Input(shape=(seq_length, 4), name="seq")

    # Stem
    x = layers.Conv1D(288, 17, padding="same", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.MaxPooling1D(3)(x)

    # Conv tower
    for f in tower_channels:
        x = layers.Conv1D(f, 5, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.MaxPooling1D(2)(x)

    # Pointwise conv -> flatten
    x = layers.Conv1D(256, 1, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Sequence embedding (bottleneck)
    seq_embed = layers.Dense(latent_dim, name="seq_embed",
                             kernel_regularizer=reg)(x)
    seq_embed = layers.BatchNormalization()(seq_embed)
    seq_embed = layers.Activation("gelu")(seq_embed)

    # Cell-wise dense head: W (latent_dim, n_cells), b (n_cells,)
    logits = layers.Dense(n_cells, name="cell_proj",
                          kernel_regularizer=reg)(seq_embed)
    output = layers.Activation("sigmoid")(logits)

    return Model(inputs, output, name="scBasset")


# ----------------------------------------------------------------------------
# Data generator (h5 random access keeps RAM bounded for 3M+ bins)
# ----------------------------------------------------------------------------

class H5Sequence:
    """Iterator over (X_batch, Y_batch) for tf.data.Dataset.from_generator.

    Reads X from seqs.h5 (rows indexed by `train_idx_in_h5`) and Y from
    the dense slice of the sparse training matrix.
    """

    def __init__(self, h5_path: Path, train_idx_in_h5: np.ndarray,
                 Y_sparse: sp.csr_matrix, row_subset: np.ndarray,
                 batch_size: int, shuffle: bool, seed: int):
        self.h5_path = str(h5_path)
        self.train_idx_in_h5 = train_idx_in_h5
        self.Y_sparse = Y_sparse
        self.row_subset = row_subset      # indices into Y_sparse (== train ordering)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(np.ceil(len(self.row_subset) / self.batch_size))

    def __call__(self):
        order = np.arange(len(self.row_subset))
        while True:
            if self.shuffle:
                self.rng.shuffle(order)
            with h5py.File(self.h5_path, "r") as h5:
                X = h5["X"]
                for b0 in range(0, len(order), self.batch_size):
                    sel = order[b0:b0 + self.batch_size]
                    train_pos = self.row_subset[sel]                 # rows of Y
                    h5_rows = self.train_idx_in_h5[train_pos]        # rows of X (in h5)
                    # h5py requires monotonically increasing fancy index
                    sort = np.argsort(h5_rows)
                    inv  = np.argsort(sort)
                    xb = X[h5_rows[sort].tolist()][inv]              # (B, L, 4) uint8
                    yb = np.asarray(self.Y_sparse[train_pos].todense())  # (B, n_cells)
                    yield xb.astype(np.float32), yb.astype(np.float32)


def main() -> int:
    ap = base_parser(__doc__ or "")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config), args.work_dir)
    m_cfg = cfg.setdefault("model", {})
    t_cfg = cfg.setdefault("train", {})

    if args.epochs is not None:     t_cfg["epochs"]     = args.epochs
    if args.batch_size is not None: t_cfg["batch_size"] = args.batch_size

    latent_dim    = int(m_cfg.get("latent_dim", 32))
    dropout_rate  = float(m_cfg.get("dropout_rate", 0.20))
    l2            = float(m_cfg.get("l2", 0.0))
    tower_channels = list(m_cfg.get("tower_channels", [288, 323, 363, 407, 456, 512]))

    epochs    = int(t_cfg.get("epochs", 500))
    patience  = int(t_cfg.get("patience", 10))
    val_frac  = float(t_cfg.get("val_frac", 0.10))
    lr        = float(t_cfg.get("learning_rate", 1e-3))
    seed      = int(t_cfg.get("random_seed", 555))
    batch_sz  = int(t_cfg.get("batch_size", 64))
    save_best = bool(t_cfg.get("save_best_only", True))

    seqs_dir  = Path(cfg["paths"]["seqs"])
    model_dir = Path(cfg["paths"]["model"])
    h5_path   = seqs_dir / "seqs.h5"
    train_npz = seqs_dir / "matrix_train.npz"
    trainable_idx_path = seqs_dir / "trainable_idx_in_h5.npy"
    for p in (h5_path, train_npz, trainable_idx_path):
        if not p.exists():
            log.error("Missing input %s -- run 02_prepare_seqs.py first", p); return 1

    log.info("TensorFlow import ...")
    import tensorflow as tf
    from tensorflow import keras
    log.info("TF %s, GPUs: %s", tf.__version__, tf.config.list_physical_devices("GPU"))
    tf.random.set_seed(seed)
    np.random.seed(seed)

    log.info("Loading training matrix %s", train_npz)
    Y = sp.load_npz(train_npz).tocsr().astype(np.uint8)
    train_idx_in_h5 = np.load(trainable_idx_path)
    assert Y.shape[0] == len(train_idx_in_h5), (
        f"Y rows {Y.shape[0]} != trainable_idx len {len(train_idx_in_h5)}"
    )
    n_cells = Y.shape[1]
    with h5py.File(h5_path, "r") as h5:
        seq_length = int(h5.attrs["seq_length"])
        n_bins_h5 = int(h5["X"].shape[0])
    log.info("Train: %d bins x %d cells  (h5 has %d total bins, seq_length=%d)",
             Y.shape[0], n_cells, n_bins_h5, seq_length)

    # Train / val split over training bins
    rng = np.random.default_rng(seed)
    n_train_bins = Y.shape[0]
    perm = rng.permutation(n_train_bins)
    n_val = max(1, int(round(val_frac * n_train_bins)))
    val_rows   = perm[:n_val]
    train_rows = perm[n_val:]
    log.info("Train bins: %d  Val bins: %d", len(train_rows), len(val_rows))

    train_gen = H5Sequence(h5_path, train_idx_in_h5, Y, train_rows,
                           batch_sz, shuffle=True,  seed=seed)
    val_gen   = H5Sequence(h5_path, train_idx_in_h5, Y, val_rows,
                           batch_sz, shuffle=False, seed=seed)

    out_sig = (
        tf.TensorSpec(shape=(None, seq_length, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, n_cells),       dtype=tf.float32),
    )
    train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=out_sig).prefetch(2)
    val_ds   = tf.data.Dataset.from_generator(val_gen,   output_signature=out_sig).prefetch(2)

    # Cap steps so an "epoch" is bounded; per-step shuffle inside generator.
    steps_per_epoch  = len(train_gen)
    validation_steps = len(val_gen)

    log.info("Building scBasset (n_cells=%d, latent=%d, tower=%s)",
             n_cells, latent_dim, tower_channels)
    model = build_scbasset(n_cells, seq_length, latent_dim,
                           tower_channels, dropout_rate, l2)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auroc", from_logits=False)],
    )
    model.summary(print_fn=log.info)

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best.weights.h5"),
            monitor="val_loss", save_best_only=save_best,
            save_weights_only=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(str(model_dir / "training.csv"), append=False),
    ]

    log.info("Training: epochs=%d steps_per_epoch=%d", epochs, steps_per_epoch)
    hist = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=cbs, verbose=2,
    )

    # Persist
    model.save(model_dir / "scbasset.keras")

    # Cell embeddings = the cell_proj layer weights (latent_dim, n_cells)
    cell_proj = model.get_layer("cell_proj")
    W, b = cell_proj.get_weights()             # W: (latent_dim, n_cells); b: (n_cells,)
    np.savez(model_dir / "cell_embeddings.npz",
             W=W.astype(np.float32), b=b.astype(np.float32))

    (model_dir / "history.json").write_text(json.dumps(
        {k: [float(x) for x in v] for k, v in hist.history.items()}, indent=2) + "\n")

    meta = {
        "seq_length":        seq_length,
        "n_cells":           int(n_cells),
        "n_train_bins":      int(Y.shape[0]),
        "n_val_bins":        int(len(val_rows)),
        "latent_dim":        latent_dim,
        "tower_channels":    tower_channels,
        "epochs_run":        int(len(hist.history.get("loss", []))),
        "best_val_loss":     float(min(hist.history.get("val_loss", [np.inf]))),
        "best_val_auroc":    float(max(hist.history.get("val_auroc", [0.0]))),
        "lr":                lr,
        "batch_size":        batch_sz,
        "weights_h5":        str(model_dir / "best.weights.h5"),
        "model_keras":       str(model_dir / "scbasset.keras"),
        "cell_embed_npz":    str(model_dir / "cell_embeddings.npz"),
    }
    (model_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Done: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
