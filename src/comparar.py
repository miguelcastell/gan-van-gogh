"""Compara todos os treinos lado a lado, com as mesmas fotos de teste.

Uso (na raiz do projeto):
    python -m src.comparar              # usa a GPU se houver
    python -m src.comparar --cpu        # para não disputar VRAM com um treino em andamento

Cada linha é um treino (o último checkpoint de cada pasta em outputs/), em ordem cronológica.
As colunas são as mesmas 8 fotos das grades por época (as 8 primeiras de testB), então dá para
comparar esta figura com as grades de qualquer época. Cada modelo roda na resolução em que
treinou; o rótulo da linha traz a resolução e o FID do metrics.json.

Saída: report/figuras/comparacao_modelos.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models import Generator
from src.utils import OUTPUTS_DIR, ROOT, get_device, last_checkpoint, load_domain, to_float, to_unit

FIG_DIR = ROOT / "report" / "figuras"
N_FOTOS = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="roda na CPU (para não disputar a GPU)")
    return p.parse_args()


def treinos():
    """Pastas de outputs/ com checkpoint, da mais antiga para a mais nova."""
    pastas = [d for d in OUTPUTS_DIR.iterdir() if any((d / "checkpoints").glob("epoch_*.pt"))]
    return sorted(pastas, key=lambda d: min(p.stat().st_mtime for p in (d / "checkpoints").glob("*.pt")))


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cpu") if args.cpu else get_device()
    linhas = []  # (rótulo, imagens (N, H, W, 3) em [0, 1])

    original = to_unit(to_float(load_domain("foto", train=False, size=256)[:N_FOTOS]))
    linhas.append(("fotos originais", original.permute(0, 2, 3, 1)))

    for pasta in treinos():
        ckpt_path = last_checkpoint(pasta)
        ckpt = torch.load(ckpt_path, map_location=device)
        size = ckpt["args"]["size"]
        G = Generator(n_blocks=ckpt["args"]["n_blocks"]).to(device).eval()
        G.load_state_dict(ckpt["G"])
        fotos = to_float(load_domain("foto", train=False, size=size)[:N_FOTOS]).to(device)
        saida = to_unit(G(fotos)).cpu().permute(0, 2, 3, 1)

        rotulo = f"{pasta.name}\n{ckpt_path.stem.replace('epoch_', 'época ')}, {size}px"
        metrics = pasta / "metrics.json"
        if metrics.exists():
            m = json.loads(metrics.read_text(encoding="utf-8"))
            if m.get("checkpoint") == ckpt_path.name:  # só se a avaliação é deste checkpoint
                rotulo += f"\nFID {m['fid']:.1f}"
        linhas.append((rotulo, saida))
        print(f"{pasta.name}: {ckpt_path.name} ({size}px)")

    fig, axes = plt.subplots(len(linhas), N_FOTOS, figsize=(2.2 * N_FOTOS, 2.3 * len(linhas)))
    for r, (rotulo, imgs) in enumerate(linhas):
        for c in range(N_FOTOS):
            ax = axes[r, c]
            ax.imshow(imgs[c].clamp(0, 1).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r, 0].set_ylabel(rotulo, fontsize=9, rotation=0, ha="right", va="center")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "comparacao_modelos.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Figura salva em {out}")


if __name__ == "__main__":
    main()
