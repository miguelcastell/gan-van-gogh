"""Avaliação de uma CycleGAN treinada: estilo (FID/KID), preservação do conteúdo,
memorização e piores saídas. Usa as fotos de teste, que o modelo nunca viu.

Uso:
    python -m src.evaluate --model cyclegan
    python -m src.evaluate --model cyclegan_sem_cycle --checkpoint epoch_040.pt
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.models import Generator
from src.utils import (OUTPUTS_DIR, ROOT, get_device, last_checkpoint, load_domain,
                       save_grid, set_seed, to_float, to_unit)

FIG_DIR = ROOT / "report" / "figuras"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="pasta em outputs/ (cyclegan, cyclegan_sem_cycle)")
    p.add_argument("--checkpoint", default=None, help="padrão: o último salvo")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


@torch.no_grad()
def translate(net, x_uint8, device, batch=16):
    """Aplica um gerador em todas as imagens (uint8) e devolve float em [-1, 1]."""
    return torch.cat([net(to_float(x_uint8[i:i + batch]).to(device))
                      for i in range(0, x_uint8.size(0), batch)])


@torch.no_grad()
def style_metrics(real, fake, device, batch=50):
    """FID e KID com Inception (torchmetrics). Imagens em [-1, 1] -> [0, 1].

    O FID fica instável com poucas centenas de imagens. O KID foi feito para esse caso:
    ele não tem o viés do FID com amostras pequenas e vem com desvio padrão.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = KernelInceptionDistance(subset_size=min(100, len(real), len(fake)),
                                  normalize=True).to(device)
    for imgs, is_real in ((real, True), (fake, False)):
        for i in range(0, imgs.size(0), batch):
            b = to_unit(imgs[i:i + batch].to(device)).clamp(0, 1)
            fid.update(b, real=is_real)
            kid.update(b, real=is_real)
    kid_mean, kid_std = kid.compute()
    return fid.compute().item(), kid_mean.item(), kid_std.item()


@torch.no_grad()
def nearest_train_distance(queries, x_train, chunk=100):
    """Distância L2 (em pixels) e índice do vizinho mais próximo no treino."""
    train_flat = x_train.flatten(1)
    dists, idxs = [], []
    for i in range(0, queries.size(0), chunk):
        d = torch.cdist(queries[i:i + chunk].flatten(1), train_flat)
        m, j = d.min(dim=1)
        dists.append(m)
        idxs.append(j)
    return torch.cat(dists), torch.cat(idxs)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    model_dir = OUTPUTS_DIR / args.model
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = model_dir / "checkpoints" / args.checkpoint if args.checkpoint else last_checkpoint(model_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    size = ckpt["args"]["size"]
    print(f"Checkpoint: {ckpt_path}")

    G = Generator(n_blocks=ckpt["args"]["n_blocks"]).to(device).eval()
    F = Generator(n_blocks=ckpt["args"]["n_blocks"]).to(device).eval()
    G.load_state_dict(ckpt["G"])
    F.load_state_dict(ckpt["F"])

    fotos_teste = load_domain("foto", train=False, size=size)
    pinturas_teste = to_float(load_domain("pintura", train=False, size=size))
    pinturas_treino = to_float(load_domain("pintura", train=True, size=size)).to(device)

    fotos = to_float(fotos_teste).to(device)
    pint = translate(G, fotos_teste, device)                       # G(foto)
    volta = torch.cat([F(pint[i:i + 16]) for i in range(0, pint.size(0), 16)])  # F(G(foto))
    results = {"model": args.model, "checkpoint": ckpt_path.name,
               "lambda_cycle": ckpt["args"]["lambda_cycle"], "n_fotos": fotos.size(0)}

    # 1) Estilo: as fotos convertidas se parecem com pinturas reais de Van Gogh?
    #    Referência: as fotos SEM conversão. Quanto o modelo aproximou as fotos das pinturas?
    results["fid"], results["kid"], results["kid_std"] = style_metrics(pinturas_teste, pint, device)
    results["fid_fotos_originais"], results["kid_fotos_originais"], _ = \
        style_metrics(pinturas_teste, fotos, device)
    print(f"FID {results['fid']:.1f} (fotos sem conversão: {results['fid_fotos_originais']:.1f}) "
          f"| KID {results['kid']:.4f} ± {results['kid_std']:.4f}")

    # 2) Conteúdo: quanto a foto muda na ida e quanto se perde na ida e volta
    erro_volta = (volta - fotos).abs().flatten(1).mean(1)
    results["cycle_l1"] = erro_volta.mean().item()
    results["mudanca_l1"] = (pint - fotos).abs().mean().item()
    print(f"Erro de ida e volta (L1) {results['cycle_l1']:.3f} | mudança foto -> pintura {results['mudanca_l1']:.3f}")

    # 3) Memorização: a saída é parecida demais com alguma pintura de treino?
    #    Referência: a distância de uma pintura de teste (real, não vista) até o treino.
    d_fake, nn_idx = nearest_train_distance(pint, pinturas_treino)
    d_ref, _ = nearest_train_distance(pinturas_teste.to(device), pinturas_treino)
    results["nn_dist_fake_mean"] = d_fake.mean().item()
    results["nn_dist_ref_mean"] = d_ref.mean().item()
    results["nn_dist_fake_min"] = d_fake.min().item()
    print(f"Dist. vizinho mais próximo: saídas {d_fake.mean():.1f} | pinturas de teste {d_ref.mean():.1f}")
    if d_ref.min() < 1e-3:
        print("Atenção: há pinturas de teste idênticas a pinturas de treino; a referência fica distorcida.")

    closest = d_fake.argsort()[:8]  # pares mais suspeitos de cópia
    pairs = torch.stack([pint[closest], pinturas_treino[nn_idx[closest]]], dim=1).flatten(0, 1)
    save_grid(pairs, FIG_DIR / f"{args.model}_vizinhos.png", nrow=2)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(d_ref.cpu(), bins=40, alpha=0.6, density=True, label="pintura de teste → treino")
    ax.hist(d_fake.cpu(), bins=40, alpha=0.6, density=True, label="saída → treino")
    ax.set_xlabel("distância L2 à pintura de treino mais próxima")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{args.model}_memorizacao.png", dpi=150)
    plt.close()

    # 4) Exemplos: as primeiras fotos e as piores (maior erro de ida e volta).
    #    Cada linha: foto | G(foto) | F(G(foto))
    def trios(idx):
        return torch.stack([fotos[idx], pint[idx], volta[idx]], dim=1).flatten(0, 1)
    save_grid(trios(torch.arange(8)), FIG_DIR / f"{args.model}_amostras.png", nrow=3)
    save_grid(trios(erro_volta.argsort(descending=True)[:8]), FIG_DIR / f"{args.model}_piores.png", nrow=3)

    out_json = model_dir / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Métricas salvas em {out_json}; figuras em {FIG_DIR}")


if __name__ == "__main__":
    main()
