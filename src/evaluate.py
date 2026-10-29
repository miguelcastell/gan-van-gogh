"""Avaliação de um gerador treinado: realismo (FID), diversidade, memorização e piores amostras.

Pré-requisito: python -m src.train_classifier

Uso:
    python -m src.evaluate --model dcgan
    python -m src.evaluate --model wgan_gp --checkpoint epoch_040.pt
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models import NZ, Classifier, Generator
from src.utils import (CLASSES, OUTPUTS_DIR, ROOT, get_device,
                       load_fashion_mnist, save_grid, set_seed, to_unit)

FIG_DIR = ROOT / "report" / "figuras"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="pasta em outputs/ (dcgan, wgan_gp)")
    p.add_argument("--checkpoint", default=None, help="padrão: o último salvo")
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--n-memo", type=int, default=1_000,
                   help="amostras usadas no teste de memorização")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


@torch.no_grad()
def generate(G, n, device, batch=500):
    return torch.cat([G(torch.randn(min(batch, n - i), NZ, device=device))
                      for i in range(0, n, batch)])


@torch.no_grad()
def compute_fid(real, fake, device, batch=200):
    """FID com Inception (torchmetrics). Imagens em [-1, 1], 1 canal -> 3 canais em [0, 1]."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for imgs, is_real in ((real, True), (fake, False)):
        for i in range(0, imgs.size(0), batch):
            fid.update(to_unit(imgs[i:i + batch]).clamp(0, 1).repeat(1, 3, 1, 1),
                       real=is_real)
    return fid.compute().item()


@torch.no_grad()
def nearest_train_distance(queries, x_train, chunk=1000):
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

    ckpts = sorted((model_dir / "checkpoints").glob("*.pt"))
    ckpt_path = model_dir / "checkpoints" / args.checkpoint if args.checkpoint else ckpts[-1]
    print(f"Checkpoint: {ckpt_path}")

    G = Generator().to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device)["G"])
    G.eval()

    clf = Classifier().to(device)
    clf.load_state_dict(torch.load(OUTPUTS_DIR / "classifier.pt", map_location=device))
    clf.eval()

    x_train, _ = load_fashion_mnist(train=True, device=device)
    x_test, _ = load_fashion_mnist(train=False, device=device)
    fake = generate(G, args.n_samples, device)
    results = {"model": args.model, "checkpoint": ckpt_path.name,
               "n_samples": args.n_samples}

    # 1) Realismo: FID contra o conjunto de teste
    results["fid"] = compute_fid(x_test, fake, device)
    print(f"FID: {results['fid']:.2f}")

    # 2) Diversidade: distribuição de classes previstas nas amostras geradas
    with torch.no_grad():
        probs = torch.cat([clf(fake[i:i + 1000]).softmax(1)
                           for i in range(0, fake.size(0), 1000)])
    conf, pred = probs.max(1)
    counts = torch.bincount(pred, minlength=10).cpu().numpy()
    dist = counts / counts.sum()
    entropy = float(-(dist[dist > 0] * np.log(dist[dist > 0])).sum() / np.log(10))
    results["class_distribution"] = dict(zip(CLASSES, dist.round(4).tolist()))
    results["class_entropy_norm"] = entropy  # 1.0 = perfeitamente uniforme
    results["mean_confidence"] = conf.mean().item()
    print(f"Entropia normalizada das classes: {entropy:.3f}")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(CLASSES, dist)
    ax.axhline(0.1, ls="--", c="gray", label="uniforme (real)")
    ax.set_ylabel("fração das amostras")
    ax.set_title(f"Distribuição de classes — {args.model}")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{args.model}_classes.png", dpi=150)
    plt.close()

    # 3) Memorização: vizinho mais próximo no treino, comparado à referência teste->treino
    d_fake, nn_idx = nearest_train_distance(fake[:args.n_memo], x_train)
    d_test, _ = nearest_train_distance(x_test[:args.n_memo], x_train)
    results["nn_dist_fake_mean"] = d_fake.mean().item()
    results["nn_dist_test_mean"] = d_test.mean().item()  # referência: imagens reais não vistas
    results["nn_dist_fake_min"] = d_fake.min().item()
    print(f"Dist. vizinho mais próximo: gerado {d_fake.mean():.2f} | teste {d_test.mean():.2f}")

    closest = d_fake.argsort()[:8]  # pares mais suspeitos de cópia
    pairs = torch.stack([fake[closest], x_train[nn_idx[closest]]], dim=1).flatten(0, 1)
    save_grid(pairs, FIG_DIR / f"{args.model}_vizinhos.png", nrow=2)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(d_test.cpu(), bins=40, alpha=0.6, label="teste → treino")
    ax.hist(d_fake.cpu(), bins=40, alpha=0.6, label="gerado → treino")
    ax.set_xlabel("distância L2 ao vizinho mais próximo")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{args.model}_memorizacao.png", dpi=150)
    plt.close()

    # 4) Exemplos ruins: menor confiança do classificador
    worst = conf.argsort()[:32]
    save_grid(fake[worst], FIG_DIR / f"{args.model}_piores.png", nrow=8)
    save_grid(fake[:64], FIG_DIR / f"{args.model}_amostras.png", nrow=8)

    out_json = model_dir / "metrics.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Métricas salvas em {out_json}; figuras em {FIG_DIR}")


if __name__ == "__main__":
    main()
