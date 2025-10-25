
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from scipy import stats  # Import the stats module from SciPy
from dataclasses import dataclass
import random

from models import construct_model
from datasets import MNISTDataRaterDataset, DataCorruptionConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class DownstreamConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128
    train_split_ratio: float = 0.8
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    keep_threshold: float = 0.75   # (unused now, kept for back-compat)
    seed: int = 42
    drop_frac: float = 0.01        # fraction to drop per batch for filtered/random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def train_one_epoch_baseline(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(loader.dataset)


def train_one_epoch_filtered(
    model,
    data_rater,
    loader,
    optimizer,
    device,
    drop_frac: float = 0.25,   # drop bottom 25% of each batch
    center: bool = False,      # optional: center scores before ranking
    min_keep: int = 1          # always keep at least this many
):
    """
    Train using only the top (1 - drop_frac) fraction by DataRater score per batch.
    Uses raw scores (no softmax), so it's not sensitive to batch size.
    """
    model.train()
    data_rater.eval()
    loss_fn = nn.CrossEntropyLoss()

    drop_frac = float(drop_frac)
    drop_frac = max(0.0, min(0.99, drop_frac))  # clamp for sanity
    min_keep = max(1, int(min_keep))

    kept_total = 0
    seen_total = 0
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B = y.size(0)
        seen_total += B

        with torch.no_grad():
            scores = data_rater(x).squeeze(-1)  # [B]
            if center:
                scores = scores - scores.mean()

        # how many to keep this batch
        keep_k = max(min_keep, B - int(B * drop_frac))
        keep_k = min(keep_k, B)  # safety

        # take top-k by raw score
        _, top_idx = torch.topk(scores, k=keep_k, largest=True, sorted=False)
        keep_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        keep_mask[top_idx] = True

        x_keep = x[keep_mask]
        y_keep = y[keep_mask]
        kept_total += y_keep.size(0)

        optimizer.zero_grad()
        logits = model(x_keep)
        loss = loss_fn(logits, y_keep)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y_keep.size(0)

    avg_loss = running_loss / max(1, kept_total)
    acceptance_rate = kept_total / max(1, seen_total)
    return avg_loss, acceptance_rate


def train_one_epoch_random_drop(
    model,
    loader,
    optimizer,
    device,
    drop_frac: float = 0.25,  # drop bottom 25% at random (no rater)
    min_keep: int = 1
):
    """
    Train keeping a random top (1 - drop_frac) fraction per batch.
    Provides a control to compare against DataRater-based filtering.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    drop_frac = float(drop_frac)
    drop_frac = max(0.0, min(0.99, drop_frac))
    min_keep = max(1, int(min_keep))

    kept_total = 0
    seen_total = 0
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B = y.size(0)
        seen_total += B

        # how many to keep this batch
        keep_k = max(min_keep, B - int(B * drop_frac))
        keep_k = min(keep_k, B)  # safety

        # pick random indices to keep
        idx = torch.randperm(B, device=x.device)[:keep_k]
        keep_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        keep_mask[idx] = True

        x_keep = x[keep_mask]
        y_keep = y[keep_mask]
        kept_total += y_keep.size(0)

        optimizer.zero_grad()
        logits = model(x_keep)
        loss = loss_fn(logits, y_keep)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y_keep.size(0)

    avg_loss = running_loss / max(1, kept_total)
    acceptance_rate = kept_total / max(1, seen_total)
    return avg_loss, acceptance_rate


def run_downstream_comparison(
    trained_data_rater: nn.Module,
    dataset_handler,   # your MNISTDataRaterDataset() instance
    config: DownstreamConfig,
    corruption_config=None  # if your dataset handler needs it for get_loaders
):
    set_seed(config.seed)

    # Build loaders
    train_loader, val_loader, test_loader = dataset_handler.get_loaders(
        config.batch_size,
        config.train_split_ratio,
        corruption_config if corruption_config is not None else DataCorruptionConfig()
    )

    device = config.device
    trained_data_rater = trained_data_rater.to(device).eval()

    # Initialize three identical CNNs (same init => fair comparison)
    base_init = construct_model('ToyCNN').to(device)
    base_init.eval()
    init_state = base_init.state_dict()

    baseline_model = construct_model('ToyCNN').to(device)
    filtered_model = construct_model('ToyCNN').to(device)
    randomdrop_model = construct_model('ToyCNN').to(device)
    baseline_model.load_state_dict(init_state)
    filtered_model.load_state_dict(init_state)
    randomdrop_model.load_state_dict(init_state)

    # Optimizers
    opt_base = optim.Adam(baseline_model.parameters(),
                          lr=config.lr, weight_decay=config.weight_decay)
    opt_filt = optim.Adam(filtered_model.parameters(),
                          lr=config.lr, weight_decay=config.weight_decay)
    opt_rand = optim.Adam(randomdrop_model.parameters(),
                          lr=config.lr, weight_decay=config.weight_decay)

    # -------- Baseline --------
    print(
        f"\n=== Baseline training (no dropping) for {config.epochs} epochs ===")
    for ep in range(1, config.epochs + 1):
        train_loss = train_one_epoch_baseline(
            baseline_model, train_loader, opt_base, device)
        val_acc = evaluate(baseline_model, val_loader, device)
        print(
            f"[Baseline][Epoch {ep}/{config.epochs}] loss={train_loss:.4f}  val_acc={val_acc:.4f}")
    baseline_test_acc = evaluate(baseline_model, test_loader, device)
    print(f"[Baseline] Test Accuracy: {baseline_test_acc:.4f}")

    # -------- DataRater Filtered --------
    print(
        f"\n=== Filtered training with DataRater (drop_frac={config.drop_frac:.2f}) for {config.epochs} epochs ===")
    for ep in range(1, config.epochs + 1):
        train_loss, acceptance = train_one_epoch_filtered(
            filtered_model, trained_data_rater, train_loader, opt_filt, device,
            drop_frac=config.drop_frac, center=True, min_keep=8
        )
        val_acc = evaluate(filtered_model, val_loader, device)
        print(
            f"[Filtered][Epoch {ep}/{config.epochs}] loss={train_loss:.4f}  val_acc={val_acc:.4f}  acceptance={acceptance*100:.1f}%")
    filtered_test_acc = evaluate(filtered_model, test_loader, device)
    print(f"[Filtered] Test Accuracy: {filtered_test_acc:.4f}")

    # -------- Random Drop (control) --------
    print(
        f"\n=== Random-drop training (drop_frac={config.drop_frac:.2f}) for {config.epochs} epochs ===")
    for ep in range(1, config.epochs + 1):
        train_loss, acceptance = train_one_epoch_random_drop(
            randomdrop_model, train_loader, opt_rand, device,
            drop_frac=config.drop_frac, min_keep=8
        )
        val_acc = evaluate(randomdrop_model, val_loader, device)
        print(
            f"[RandomDrop][Epoch {ep}/{config.epochs}] loss={train_loss:.4f}  val_acc={val_acc:.4f}  acceptance={acceptance*100:.1f}%")
    randomdrop_test_acc = evaluate(randomdrop_model, test_loader, device)
    print(f"[RandomDrop] Test Accuracy: {randomdrop_test_acc:.4f}")

    print("\n=== Summary ===")
    print(f"Baseline  Test Acc : {baseline_test_acc:.4f}")
    print(f"Filtered  Test Acc : {filtered_test_acc:.4f}")
    print(f"RandomDrop Test Acc: {randomdrop_test_acc:.4f}")
    return {
        "baseline_test_acc": baseline_test_acc,
        "filtered_test_acc": filtered_test_acc,
        "randomdrop_test_acc": randomdrop_test_acc,
    }


def run_trials(
    trained_data_rater,
    dataset_handler,
    base_config: DownstreamConfig,
    corruption_config=None,
    n_trials: int = 5
):
    """
    Runs n_trials of the downstream comparison, reseeding each time with base_config.seed + i.
    Collects test accuracies and reports mean ± std.
    """
    baseline_accs = []
    filtered_accs = []
    randomdrop_accs = []

    for i in range(n_trials):
        cfg = DownstreamConfig(
            device=base_config.device,
            batch_size=base_config.batch_size,
            train_split_ratio=base_config.train_split_ratio,
            epochs=base_config.epochs,
            lr=base_config.lr,
            weight_decay=base_config.weight_decay,
            keep_threshold=base_config.keep_threshold,
            seed=base_config.seed + i,   # reseed per trial
            drop_frac=base_config.drop_frac
        )

        print(
            f"\n========== Trial {i+1}/{n_trials} (seed={cfg.seed}) ==========")
        out = run_downstream_comparison(
            trained_data_rater=trained_data_rater,
            dataset_handler=dataset_handler,
            config=cfg,
            corruption_config=corruption_config
        )
        baseline_accs.append(out["baseline_test_acc"])
        filtered_accs.append(out["filtered_test_acc"])
        randomdrop_accs.append(out["randomdrop_test_acc"])

    baseline_accs = np.array(baseline_accs, dtype=float)
    filtered_accs = np.array(filtered_accs, dtype=float)
    randomdrop_accs = np.array(randomdrop_accs, dtype=float)

    def _stat(a):
        return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0

    b_mean, b_std = _stat(baseline_accs)
    f_mean, f_std = _stat(filtered_accs)
    r_mean, r_std = _stat(randomdrop_accs)

    print("\n=========== Final Summary over Trials ===========")
    print(f"Baseline    : {b_mean:.4f} ± {b_std:.4f} (n={n_trials})")
    print(f"Filtered    : {f_mean:.4f} ± {f_std:.4f} (n={n_trials})")
    print(f"Random-Drop : {r_mean:.4f} ± {r_std:.4f} (n={n_trials})")

    return {
        "trials": n_trials,
        "baseline_test_accs": baseline_accs.tolist(),
        "filtered_test_accs": filtered_accs.tolist(),
        "randomdrop_test_accs": randomdrop_accs.tolist(),
        "summary": {
            "baseline_mean": b_mean, "baseline_std": b_std,
            "filtered_mean": f_mean, "filtered_std": f_std,
            "randomdrop_mean": r_mean, "randomdrop_std": r_std,
        }
    }


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def load_data_rater_from_checkpoint(checkpoint_path, device='cpu'):
        model = construct_model('DataRater').to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    checkpoint_path = "mnist_20250920_1037_a11efc10/data_rater.pt"
    trained_data_rater = load_data_rater_from_checkpoint(
        checkpoint_path, DEVICE)

    dataset_handler = MNISTDataRaterDataset()
    corruption_cfg = DataCorruptionConfig()

    base_cfg = DownstreamConfig(
        device=DEVICE,
        batch_size=128,
        epochs=1,        # bump to 5–10 for a more stable comparison
        lr=1e-3,
        drop_frac=0.10
    )

    _ = run_trials(
        trained_data_rater=trained_data_rater,
        dataset_handler=dataset_handler,
        base_config=base_cfg,
        corruption_config=corruption_cfg,
        n_trials=5
    )
