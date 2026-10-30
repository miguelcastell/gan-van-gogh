"""Funções compartilhadas: seeds, dados, logs, amostras e checkpoints."""
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

DATASET_DIR = DATA_DIR / "vangogh2photo"
DATASET_URL = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip"

# Os nomes A e B vêm do dataset: A = pinturas de Van Gogh, B = fotos.
DOMAINS = {"pintura": "A", "foto": "B"}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_domain(domain: str, train: bool = True, size: int = 128) -> torch.Tensor:
    """Carrega todas as imagens de um domínio como tensor uint8 (N, 3, size, size).

    domain: "pintura" ou "foto". Ver ADR 0003: tudo em memória, em uint8, e a conversão
    para [-1, 1] só acontece no batch (`to_float`). Na primeira vez as imagens são lidas
    e redimensionadas; depois o resultado é lido de um cache .pt em data/.
    """
    folder = ("train" if train else "test") + DOMAINS[domain]
    cache = DATA_DIR / f"vangogh2photo_{size}" / f"{folder}.pt"
    if cache.exists():
        return torch.load(cache)

    paths = sorted((DATASET_DIR / folder).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem em {DATASET_DIR / folder}. "
                                "Rode antes: python -m src.download_data")
    x = torch.stack([load_image(p, size) for p in paths])
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(x, cache)
    return x


def load_image(path: Path, size: int | None = None) -> torch.Tensor:
    """Lê uma imagem como tensor uint8 (3, H, W). Com `size`, redimensiona para size x size."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    return pil_to_tensor(img)


def to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1], a mesma faixa da saída `tanh` do gerador."""
    return x.float().div(127.5).sub(1.0)


def random_batch(x: torch.Tensor, batch_size: int, device: torch.device,
                 flip: bool = True, crop: int | None = None) -> torch.Tensor:
    """Sorteia um batch, converte para [-1, 1] e espelha metade das imagens na horizontal.

    crop: recorta de cada imagem um quadrado crop x crop numa posição sorteada. Assim a mesma
    pintura chega ao discriminador um pouco diferente a cada vez, e fica mais difícil decorá-la.
    Os recortes são feitos na hora, em memória: nenhuma imagem nova é salva em disco.
    """
    idx = torch.randint(0, x.size(0), (batch_size,))
    batch = x[idx]
    if crop is not None and crop < batch.size(-1):
        tops = torch.randint(0, batch.size(-2) - crop + 1, (batch_size,))
        lefts = torch.randint(0, batch.size(-1) - crop + 1, (batch_size,))
        batch = torch.stack([img[:, t:t + crop, l:l + crop] for img, t, l in zip(batch, tops, lefts)])
    batch = to_float(batch.to(device))
    if flip:
        mask = torch.rand(batch_size, device=device) < 0.5
        batch[mask] = batch[mask].flip(-1)
    return batch


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]."""
    return (x + 1) / 2


def save_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_unit(images).clamp(0, 1), path, nrow=nrow, padding=2)


class CSVLogger:
    def __init__(self, path: Path, fields: list[str], append: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fields = fields
        if not (append and path.exists()):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def last_checkpoint(model_dir: Path) -> Path:
    ckpts = sorted((model_dir / "checkpoints").glob("*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"Nenhum checkpoint em {model_dir / 'checkpoints'}")
    return ckpts[-1]


def make_gif(samples_dir: Path, out_path: Path, fps: int = 4) -> None:
    import imageio.v2 as imageio
    frames = [imageio.imread(p) for p in sorted(samples_dir.glob("epoch_*.png"))]
    if frames:
        imageio.mimsave(out_path, frames, duration=1 / fps, loop=0)
