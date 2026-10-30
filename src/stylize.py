"""Pinta o céu de uma foto no estilo de Van Gogh.

Uso (na raiz do projeto):
    python -m src.stylize minha_foto.jpg
    python -m src.stylize minha_foto.jpg --model cyclegan_sem_cycle
    python -m src.stylize minha_foto.jpg --sem-mascara      # estiliza a foto inteira

Saída em outputs/stylize/: o resultado e um painel original | máscara | estilizada | resultado.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf

from src.models import Generator
from src.sky import composite, sky_mask
from src.utils import (OUTPUTS_DIR, get_device, last_checkpoint, load_image, save_grid,
                       to_float, to_unit)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("foto", type=Path)
    p.add_argument("--model", default="cyclegan", help="pasta em outputs/")
    p.add_argument("--checkpoint", default=None, help="padrão: o último salvo")
    p.add_argument("--size", type=int, default=256,
                   help="lado menor da imagem que entra no gerador. O gerador treinou em 128: "
                        "valores maiores deixam as pinceladas menores em relação à cena")
    p.add_argument("--max-side", type=int, default=1024, help="reduz fotos maiores que isso")
    p.add_argument("--sem-mascara", action="store_true")
    return p.parse_args()


def resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return nnf.interpolate(x, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()
    model_dir = OUTPUTS_DIR / args.model
    ckpt_path = model_dir / "checkpoints" / args.checkpoint if args.checkpoint else last_checkpoint(model_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Checkpoint: {ckpt_path}")

    G = Generator(n_blocks=ckpt["args"]["n_blocks"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    img = load_image(args.foto)
    h, w = img.shape[1:]
    if max(h, w) > args.max_side:
        s = args.max_side / max(h, w)
        h, w = int(h * s), int(w * s)
        img = (to_unit(resize(to_float(img)[None], (h, w))[0]) * 255).round().byte()
    x = to_float(img)[None].to(device)

    # O gerador roda numa versão menor da foto (lados múltiplos de 4) e o resultado volta
    # ao tamanho original. A foto original não perde resolução fora do céu.
    s = args.size / min(h, w)
    small = (max(4, round(h * s / 4) * 4), max(4, round(w * s / 4) * 4))
    stylized = resize(G(resize(x, small)), (h, w))

    if args.sem_mascara:
        mask = torch.ones(1, h, w, device=device)
    else:
        mask = sky_mask(img, device)
    result = composite(x, stylized, mask[None])

    out = OUTPUTS_DIR / "stylize"
    stem = f"{args.foto.stem}_{args.model}"
    save_grid(result, out / f"{stem}.png", nrow=1)

    paineis = [("original", x), ("máscara do céu", mask[None] * 2 - 1),
               ("estilizada", stylized), ("resultado", result)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5 * h / w))
    for ax, (titulo, im) in zip(axes, paineis):
        im = to_unit(im[0]).clamp(0, 1).permute(1, 2, 0).cpu()
        ax.imshow(im.squeeze(-1) if im.size(-1) == 1 else im, cmap="gray")
        ax.set_title(titulo)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / f"{stem}_painel.png", dpi=120)
    plt.close()
    print(f"Salvo em {out / stem}.png e {stem}_painel.png")


if __name__ == "__main__":
    main()
