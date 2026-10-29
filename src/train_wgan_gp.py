"""Treino da WGAN-GP no Fashion-MNIST (mesmo gerador do DCGAN).

Uso (na raiz do projeto):
    python -m src.train_wgan_gp --epochs 40
"""
import argparse
import time

import torch

from src.models import NZ, Critic, Generator, weights_init
from src.utils import (OUTPUTS_DIR, CSVLogger, get_device, iterate_minibatches,
                       load_fashion_mnist, make_gif, save_checkpoint, save_grid,
                       set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="wgan_gp")
    return p.parse_args()


def gradient_penalty(C, real, fake):
    eps = torch.rand(real.size(0), 1, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grads = torch.autograd.grad(C(interp).sum(), interp, create_graph=True)[0]
    return ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    out = OUTPUTS_DIR / args.out
    print(f"Dispositivo: {device} | saída: {out}")

    x_train, _ = load_fashion_mnist(train=True, device=device)

    G = Generator().to(device)
    C = Critic().to(device)
    G.apply(weights_init)
    C.apply(weights_init)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_c = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.0, 0.9))

    fixed_z = torch.randn(64, NZ, device=device)
    logger = CSVLogger(out / "logs.csv",
                       ["epoch", "step", "loss_c", "loss_g", "wasserstein", "gp",
                        "c_real", "c_fake"])

    step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        G.train()
        # Uma "época" = uma passada pelos dados reais no crítico;
        # o gerador é atualizado a cada n_critic minibatches.
        for i, real in enumerate(iterate_minibatches(x_train, args.batch_size)):
            b = real.size(0)
            fake = G(torch.randn(b, NZ, device=device)).detach()

            opt_c.zero_grad(set_to_none=True)
            c_real, c_fake = C(real).mean(), C(fake).mean()
            gp = gradient_penalty(C, real, fake)
            loss_c = c_fake - c_real + args.lambda_gp * gp
            loss_c.backward()
            opt_c.step()

            if (i + 1) % args.n_critic != 0:
                continue

            opt_g.zero_grad(set_to_none=True)
            loss_g = -C(G(torch.randn(b, NZ, device=device))).mean()
            loss_g.backward()
            opt_g.step()

            if step % 10 == 0:
                logger.log({
                    "epoch": epoch, "step": step,
                    "loss_c": loss_c.item(), "loss_g": loss_g.item(),
                    "wasserstein": (c_real - c_fake).item(), "gp": gp.item(),
                    "c_real": c_real.item(), "c_fake": c_fake.item(),
                })
            step += 1

        G.eval()
        with torch.no_grad():
            save_grid(G(fixed_z), out / "samples" / f"epoch_{epoch:03d}.png")
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(out / "checkpoints" / f"epoch_{epoch:03d}.pt",
                            G=G.state_dict(), C=C.state_dict(),
                            opt_g=opt_g.state_dict(), opt_c=opt_c.state_dict(),
                            epoch=epoch, args=vars(args))
        print(f"Época {epoch:3d}/{args.epochs} | loss_C {loss_c.item():.3f} "
              f"| W {(c_real - c_fake).item():.3f} | {time.time() - t0:.1f}s")

    make_gif(out / "samples", out / "evolucao.gif")
    print("Treino concluído.")


if __name__ == "__main__":
    main()
