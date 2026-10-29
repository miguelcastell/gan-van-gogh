"""Funções compartilhadas: seeds, dados, logs, amostras e checkpoints."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fashion_mnist(train: bool = True, device: torch.device | None = None):
    """Carrega o Fashion-MNIST inteiro como tensor (N, 1, 28, 28) em [-1, 1].

    A base cabe na RAM/VRAM, então evitamos DataLoader e gargalo de disco.
    """
    ds = datasets.FashionMNIST(DATA_DIR, train=train, download=True)
    x = ds.data.float().div(127.5).sub(1.0).unsqueeze(1)
    y = ds.targets.clone()
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


def iterate_minibatches(x: torch.Tensor, batch_size: int, drop_last: bool = True):
    idx = torch.randperm(x.size(0), device=x.device)
    stop = x.size(0) - batch_size + 1 if drop_last else x.size(0)
    for i in range(0, stop, batch_size):
        yield x[idx[i:i + batch_size]]


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (x + 1) / 2


def save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_unit(images).clamp(0, 1), path, nrow=nrow, padding=2)


class CSVLogger:
    def __init__(self, path: Path, fields: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fields = fields
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def make_gif(samples_dir: Path, out_path: Path, fps: int = 4) -> None:
    import imageio.v2 as imageio
    frames = [imageio.imread(p) for p in sorted(samples_dir.glob("epoch_*.png"))]
    if frames:
        imageio.mimsave(out_path, frames, duration=1 / fps, loop=0)
