"""Conv1D training on encrypted data using Concrete ML.

Trains a Conv1D filter on encrypted data using FHE. The architecture is:
    Conv1D -> avg pool -> sigmoid -> BCE loss

The plain PyTorch baseline uses torch.nn.Conv1d with standard autograd.
The FHE version uses the same architecture but with an explicit backward
pass inside forward() (required for FHE compilation).

Usage:
    python example/cnn1d_fhe_training.py
"""

import itertools
import time

import numpy as np
import torch
import torch.nn as nn
from concrete import fhe
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from concrete.ml.torch.compile import _compile_torch_or_onnx_model


# ---------------------------------------------------------------------------
# 1. Plain PyTorch model using nn.Conv1d (baseline)
# ---------------------------------------------------------------------------
class PlainConv1DModel(nn.Module):
    """Standard Conv1D -> AvgPool -> Sigmoid for binary classification."""

    def __init__(self, seq_len: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, 1, seq_len)
        out = self.conv(x)       # (batch, 1, out_len)
        out = self.pool(out)     # (batch, 1, 1)
        out = out.squeeze(-1)    # (batch, 1)
        return torch.sigmoid(out)


def train_plain_pytorch(X, y, seq_len, kernel_size, batch_size, n_epochs, lr, init_kernel,
                        init_bias):
    """Train with nn.Conv1d + autograd + BCE loss."""
    model = PlainConv1DModel(seq_len, kernel_size).double()

    # Initialize with the same weights as FHE version
    with torch.no_grad():
        # nn.Conv1d weight shape: (out_channels=1, in_channels=1, kernel_size)
        model.conv.weight.copy_(torch.tensor(init_kernel.reshape(1, 1, kernel_size),
                                             dtype=torch.float64))
        model.conv.bias.copy_(torch.tensor(init_bias.flatten(), dtype=torch.float64))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    n_samples = X.shape[0]
    batches_per_epoch = n_samples // batch_size

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        X_s, y_s = X[perm], y[perm]

        for i in range(batches_per_epoch):
            s = i * batch_size
            xb = torch.tensor(X_s[s:s + batch_size].reshape(batch_size, 1, seq_len),
                              dtype=torch.float64)
            yb = torch.tensor(y_s[s:s + batch_size].reshape(batch_size, 1),
                              dtype=torch.float64)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            X_all = torch.tensor(X.reshape(n_samples, 1, seq_len), dtype=torch.float64)
            preds = (model(X_all).squeeze() > 0.5).numpy().astype(int)
            acc = accuracy_score(y, preds)
            print(f"  Epoch {epoch + 1}/{n_epochs}: accuracy = {acc * 100:.2f}%")

    # Extract final weights
    final_kernel = model.conv.weight.detach().numpy().flatten()
    final_bias = model.conv.bias.detach().numpy().flatten()

    with torch.no_grad():
        X_all = torch.tensor(X.reshape(n_samples, 1, seq_len), dtype=torch.float64)
        preds = (model(X_all).squeeze() > 0.5).numpy().astype(int)

    return final_kernel, final_bias, preds


# ---------------------------------------------------------------------------
# 2. FHE-compatible Conv1D training module (explicit backward in forward)
# ---------------------------------------------------------------------------
class Conv1DTraining(torch.nn.Module):
    """Conv1D training with explicit backward pass in forward().

    Architecture: Conv1D -> Global Avg Pool -> Sigmoid -> BCE loss.
    All gradients computed manually (no autograd).

    Tensor layout: (1, batch_size, dim) following Concrete ML convention.
    """

    def __init__(self, kernel_size: int, seq_len: int, learning_rate: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.out_len = seq_len - kernel_size + 1

    def forward(
        self,
        x: torch.Tensor,       # (1, batch_size, seq_len)
        y: torch.Tensor,       # (1, batch_size, 1)
        kernel: torch.Tensor,  # (1, kernel_size, 1)
        bias: torch.Tensor,    # (1, 1, 1)
    ):
        batch_size = x.size(1)

        # ---- FORWARD: Conv1D via patches -> global avg pool -> sigmoid ----
        pooled_sum = None
        for i in range(self.out_len):
            patch_i = x[:, :, i:i + self.kernel_size]  # (1, batch_size, kernel_size)
            conv_i = patch_i @ kernel + bias            # (1, batch_size, 1)
            if pooled_sum is None:
                pooled_sum = conv_i
            else:
                pooled_sum = pooled_sum + conv_i

        logits = pooled_sum / self.out_len  # (1, batch_size, 1)
        output = torch.sigmoid(logits)      # (1, batch_size, 1)

        # ---- BACKWARD: d(BCE)/d(logits) = output - y ----
        d_logits = output - y
        d_conv = d_logits / self.out_len

        d_kernel = torch.zeros_like(kernel)
        d_bias = torch.zeros_like(bias)

        for i in range(self.out_len):
            patch_i = x[:, :, i:i + self.kernel_size]
            d_kernel = d_kernel + patch_i.transpose(1, 2) @ d_conv / batch_size
            d_bias = d_bias + d_conv.sum(dim=1, keepdim=True) / batch_size

        # ---- SGD UPDATE ----
        kernel = kernel - self.learning_rate * d_kernel
        bias = bias - self.learning_rate * d_bias

        return kernel, bias


# ---------------------------------------------------------------------------
# 3. Prediction using nn.Conv1d (for evaluating FHE-trained weights)
# ---------------------------------------------------------------------------
def predict_with_nn_conv1d(X, kernel_weights, bias_value, kernel_size):
    """Create an nn.Conv1d with the given weights and predict."""
    n_samples, seq_len = X.shape
    model = PlainConv1DModel(seq_len, kernel_size).double()
    with torch.no_grad():
        model.conv.weight.copy_(torch.tensor(kernel_weights.reshape(1, 1, kernel_size),
                                             dtype=torch.float64))
        model.conv.bias.copy_(torch.tensor(bias_value.flatten(), dtype=torch.float64))
        X_t = torch.tensor(X.reshape(n_samples, 1, seq_len), dtype=torch.float64)
        preds = (model(X_t).squeeze() > 0.5).numpy().astype(int)
    return preds


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)

    # ---- Dataset ----
    n_samples = 200
    seq_len = 4
    X, y = make_classification(n_samples=n_samples, n_features=seq_len, n_informative=3,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=2.0, random_state=42)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X).astype(np.float64)
    y = y.astype(np.float64)

    # ---- Hyperparameters ----
    kernel_size = 3
    batch_size = 8
    lr = 1.0
    n_epochs = 3
    param_range = (-1.0, 1.0)
    n_bits = 6

    # ---- Shared initial weights ----
    init_kernel = np.random.uniform(*param_range, size=(kernel_size,))
    init_bias = np.random.uniform(*param_range, size=(1,))

    # ==================================================================
    # STEP 1: Plain PyTorch training with nn.Conv1d
    # ==================================================================
    print("=" * 60)
    print("STEP 1: Plain PyTorch training (nn.Conv1d + autograd)")
    print("=" * 60)

    np.random.seed(42)  # same shuffling
    plain_kernel, plain_bias, plain_preds = train_plain_pytorch(
        X, y, seq_len, kernel_size, batch_size, n_epochs, lr, init_kernel, init_bias
    )
    plain_acc = accuracy_score(y, plain_preds)
    print(f"  Final kernel: {plain_kernel}")
    print(f"  Final bias:   {plain_bias}\n")

    # ==================================================================
    # STEP 2: Compile FHE Conv1D circuit
    # ==================================================================
    print("=" * 60)
    print("STEP 2: Compiling Conv1DTraining to FHE circuit")
    print("=" * 60)

    trainer = Conv1DTraining(kernel_size, seq_len, learning_rate=lr)

    x_min, x_max = X.min(axis=0), X.max(axis=0)
    combinations = list(itertools.product(
        [0.0, 1.0],
        [x_min, x_max, np.zeros(seq_len)],
        [param_range[0], param_range[1]],
        [param_range[0], param_range[1]],
    ))
    cs = len(combinations)

    x_c = np.empty((cs, batch_size, seq_len))
    y_c = np.empty((cs, batch_size, 1))
    k_c = np.empty((cs, kernel_size, 1))
    b_c = np.empty((cs, 1, 1))

    for idx, (label, xv, kv, bv) in enumerate(combinations):
        x_c[idx] = xv
        y_c[idx] = label
        k_c[idx] = kv
        b_c[idx] = bv

    configuration = fhe.Configuration()
    configuration.composable = True
    composition_mapping = {0: 2, 1: 3}

    print("Compiling...")
    start = time.time()
    qm = _compile_torch_or_onnx_model(
        trainer,
        (x_c, y_c, k_c, b_c),
        n_bits=n_bits,
        rounding_threshold_bits=7,
        p_error=0.01,
        configuration=configuration,
        reduce_sum_copy=True,
        composition_mapping=composition_mapping,
    )
    print(f"Compilation done in {time.time() - start:.1f}s\n")

    # ==================================================================
    # STEP 3: FHE training — simulate and execute modes
    # ==================================================================
    batches_per_epoch = n_samples // batch_size
    results = {}

    for mode_label, fhe_mode, mode_epochs in [
        ("simulate", "simulate", n_epochs),
        ("execute (real FHE encryption)", "execute", 1),  # 1 epoch — execute is slow
    ]:
        print("=" * 60)
        print(f"STEP 3: FHE training — {mode_label}")
        print("=" * 60)

        fhe_kernel = init_kernel.reshape(1, kernel_size, 1).copy()
        fhe_bias = init_bias.reshape(1, 1, 1).copy()

        if fhe_mode == "execute":
            # Generate encryption keys
            print("  Generating encryption keys...")
            key_start = time.time()
            qm.fhe_circuit.keygen()
            print(f"  Keygen done in {time.time() - key_start:.1f}s")

        np.random.seed(42)  # same shuffling as plain PyTorch
        total_time = 0.0
        for epoch in range(mode_epochs):
            perm = np.random.permutation(n_samples)
            X_s, y_s = X[perm], y[perm]
            for i in range(batches_per_epoch):
                s = i * batch_size
                xb = X_s[s:s + batch_size].reshape(1, batch_size, seq_len)
                yb = y_s[s:s + batch_size].reshape(1, batch_size, 1)

                # Quantize
                qi = qm.quantize_input(xb, yb, fhe_kernel, fhe_bias)

                if fhe_mode == "execute":
                    # Encrypt -> run on encrypted data -> decrypt
                    batch_start = time.time()
                    encrypted_inputs = qm.fhe_circuit.encrypt(*qi)
                    encrypted_outputs = qm.fhe_circuit.run(*encrypted_inputs)
                    qo = qm.fhe_circuit.decrypt(*encrypted_outputs)
                    batch_time = time.time() - batch_start
                    total_time += batch_time
                    if (i + 1) % 5 == 0 or i == 0:
                        print(f"    Batch {i + 1}/{batches_per_epoch}: "
                              f"{batch_time:.2f}s per batch")
                else:
                    qo = qm.quantized_forward(*qi, fhe=fhe_mode)

                fhe_kernel, fhe_bias = qm.dequantize_output(*qo)

            preds = predict_with_nn_conv1d(X, fhe_kernel.flatten(), fhe_bias.flatten(),
                                           kernel_size)
            acc = accuracy_score(y, preds)
            print(f"  Epoch {epoch + 1}/{mode_epochs}: accuracy = {acc * 100:.2f}%")

        if fhe_mode == "execute":
            print(f"  Total FHE execution time: {total_time:.1f}s "
                  f"({total_time / batches_per_epoch:.2f}s/batch)")

        results[fhe_mode] = {
            "acc": accuracy_score(y, predict_with_nn_conv1d(
                X, fhe_kernel.flatten(), fhe_bias.flatten(), kernel_size)),
            "kernel": fhe_kernel.flatten().copy(),
            "bias": fhe_bias.flatten().copy(),
        }
        print(f"  Final kernel: {fhe_kernel.flatten()}")
        print(f"  Final bias:   {fhe_bias.flatten()}\n")

    # ==================================================================
    # STEP 4: Compare all results
    # ==================================================================
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'Method':<35} {'Accuracy':>10}  {'Epochs':>6}")
    print(f"  {'-'*35} {'-'*10}  {'-'*6}")
    print(f"  {'nn.Conv1d (plain PyTorch)':<35} {plain_acc*100:>9.2f}%  {n_epochs:>6}")
    print(f"  {'FHE simulate':<35} {results['simulate']['acc']*100:>9.2f}%  {n_epochs:>6}")
    print(f"  {'FHE execute (real encryption)':<35} {results['execute']['acc']*100:>9.2f}%  {1:>6}")
    print()
    print(f"  {'Method':<35} {'Kernel'}")
    print(f"  {'-'*35} {'-'*40}")
    print(f"  {'nn.Conv1d (plain PyTorch)':<35} {plain_kernel}")
    print(f"  {'FHE simulate':<35} {results['simulate']['kernel']}")
    print(f"  {'FHE execute (real encryption)':<35} {results['execute']['kernel']}")


if __name__ == "__main__":
    main()
