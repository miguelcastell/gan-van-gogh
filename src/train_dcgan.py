"""Treino do DCGAN no Fashion-MNIST.

Uso (na raiz do projeto):
    python -m src.train_dcgan --epochs 40
    python -m src.train_dcgan --epochs 2      # teste rápido
"""
import argparse
import time

import torch
import torch.nn as nn

from src.models import NZ, Discriminator, Generator, weights_init
from src.utils import (OUTPUTS_DIR, CSVLogger, get_device, iterate_minibatches,
                       load_fashion_mnist, make_gif, save_checkpoint, save_grid,
                       set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="ex.: 0.1 usa rótulo 0.9 para reais (ajuda se D vencer cedo)")
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="dcgan")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out = OUTPUTS_DIR / args.out
    print(f"Dispositivo: {device} | saída: {out}")

    x_train, _ = load_fashion_mnist(train=True, device=device)

    G = Generator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, NZ, device=device)  # mesmo ruído em todas as épocas
    logger = CSVLogger(out / "logs.csv",
                       ["epoch", "step", "loss_d", "loss_g", "d_x", "d_g_z"])
    real_label = 1.0 - args.label_smoothing

    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        G.train()
        for real in iterate_minibatches(x_train, args.batch_size):
            b = real.size(0)
            z = torch.randn(b, NZ, device=device)
            fake = G(z)

            # --- Discriminador: max log D(x) + log(1 - D(G(z)))
            opt_d.zero_grad(set_to_none=True)
            out_real = D(real)
            out_fake = D(fake.detach())
            loss_d = (bce(out_real, torch.full_like(out_real, real_label))
                      + bce(out_fake, torch.zeros_like(out_fake)))
            loss_d.backward()
            opt_d.step()

            # --- Gerador: max log D(G(z)) (loss não saturante)
            opt_g.zero_grad(set_to_none=True)
            out_fake_g = D(fake)
            loss_g = bce(out_fake_g, torch.ones_like(out_fake_g))
            loss_g.backward()
            opt_g.step()

            if step % 50 == 0:
                logger.log({
                    "epoch": epoch, "step": step,
                    "loss_d": loss_d.item(), "loss_g": loss_g.item(),
                    "d_x": torch.sigmoid(out_real).mean().item(),
                    "d_g_z": torch.sigmoid(out_fake).mean().item(),
                })
            step += 1

        G.eval()
        with torch.no_grad():
            save_grid(G(fixed_z), out / "samples" / f"epoch_{epoch:03d}.png")
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(out / "checkpoints" / f"epoch_{epoch:03d}.pt",
                            G=G.state_dict(), D=D.state_dict(),
                            opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(),
                            epoch=epoch, args=vars(args))
        print(f"Época {epoch:3d}/{args.epochs} | loss_D {loss_d.item():.3f} "
              f"| loss_G {loss_g.item():.3f} | {time.time() - t0:.1f}s")

    make_gif(out / "samples", out / "evolucao.gif")
    print("Treino concluído.")


if __name__ == "__main__":
    main()
