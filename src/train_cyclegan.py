"""Treino da CycleGAN foto <-> Van Gogh.

G: foto -> pintura, julgado por D_pintura
F: pintura -> foto, julgado por D_foto

Uso (na raiz do projeto):
    python -m src.train_cyclegan --epochs 1 --photos 300    # teste rápido, mede o tempo
    python -m src.train_cyclegan --epochs 40
    python -m src.train_cyclegan --epochs 40 --sem-cycle    # comparação: sem cycle loss
    python -m src.train_cyclegan --resume outputs/cyclegan/checkpoints/epoch_010.pt
"""
import argparse
import random
import time

import torch
import torch.nn as nn

from src.models import Generator, PatchDiscriminator, weights_init
from src.utils import (OUTPUTS_DIR, CSVLogger, get_device, load_domain, make_gif,
                       random_batch, save_checkpoint, save_grid, set_seed, to_float)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--decay-start", type=int, default=None,
                   help="época em que o lr começa a cair até 0 (padrão: metade de --epochs)")
    p.add_argument("--photos", type=int, default=None,
                   help="usar só N fotos de treino (padrão: todas). Uma época = N passos")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=5.0)
    p.add_argument("--sem-cycle", action="store_true",
                   help="atalho para --lambda-cycle 0 (a comparação do projeto)")
    p.add_argument("--pool-size", type=int, default=50)
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--amp", action="store_true",
                   help="precisão mista bf16: cerca de 30%% mais rápido e usa menos VRAM")
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", default=None, help="checkpoint para continuar o treino")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="pasta em outputs/ (padrão: cyclegan ou cyclegan_sem_cycle)")
    args = p.parse_args()
    if args.sem_cycle:
        args.lambda_cycle = 0.0
    if args.out is None:
        args.out = "cyclegan_sem_cycle" if args.sem_cycle else "cyclegan"
    if args.decay_start is None:
        args.decay_start = args.epochs // 2
    return args


class ImagePool:
    """Guarda imagens falsas antigas. O discriminador recebe uma mistura de novas e antigas,
    para não reagir só à última "moda" do gerador."""

    def __init__(self, size: int = 50):
        self.size = size
        self.images = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() < 0.5:
                # devolve uma antiga e guarda a nova no lugar dela
                i = random.randrange(self.size)
                out.append(self.images[i])
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    # As imagens têm sempre o mesmo tamanho, então o cuDNN pode testar os algoritmos de
    # convolução uma vez e usar o mais rápido. Aqui isso mais que dobra a velocidade.
    torch.backends.cudnn.benchmark = True
    # --amp: as contas dentro de `with amp():` usam bfloat16 (16 bits) em vez de float32
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp)
    out = OUTPUTS_DIR / args.out
    print(f"Dispositivo: {device} | saída: {out}")
    print(f"λ_cycle {args.lambda_cycle} | λ_identity {args.lambda_identity}")

    pinturas = load_domain("pintura", train=True, size=args.size)
    fotos = load_domain("foto", train=True, size=args.size)
    if args.photos is not None:
        fotos = fotos[torch.randperm(fotos.size(0))[:args.photos]]
    steps_per_epoch = fotos.size(0) // args.batch_size
    print(f"Pinturas: {pinturas.size(0)} | fotos: {fotos.size(0)} | passos por época: {steps_per_epoch}")

    G = Generator(n_blocks=args.n_blocks).to(device)   # foto -> pintura
    F = Generator(n_blocks=args.n_blocks).to(device)   # pintura -> foto
    D_pintura = PatchDiscriminator().to(device)
    D_foto = PatchDiscriminator().to(device)
    for net in (G, F, D_pintura, D_foto):
        net.apply(weights_init)

    betas = (args.beta1, 0.999)
    # G e F são otimizados juntos, porque a cycle loss depende dos dois
    opt_g = torch.optim.Adam(list(G.parameters()) + list(F.parameters()), lr=args.lr, betas=betas)
    opt_d = torch.optim.Adam(list(D_pintura.parameters()) + list(D_foto.parameters()),
                             lr=args.lr, betas=betas)

    # lr constante até decay_start, depois cai em linha reta em direção a 0 na última época
    def lr_factor(epoch_idx):
        return 1.0 - max(0, epoch_idx - args.decay_start) / max(1, args.epochs - args.decay_start)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_factor)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_factor)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        for name, net in (("G", G), ("F", F), ("D_pintura", D_pintura), ("D_foto", D_foto)):
            net.load_state_dict(ckpt[name])
        for name, obj in (("opt_g", opt_g), ("opt_d", opt_d),
                          ("sched_g", sched_g), ("sched_d", sched_d)):
            obj.load_state_dict(ckpt[name])
        start_epoch = ckpt["epoch"] + 1
        print(f"Continuando da época {start_epoch}")

    mse = nn.MSELoss()   # LSGAN: erro quadrático até o alvo (1 = real, 0 = falsa)
    l1 = nn.L1Loss()     # cycle e identity: diferença pixel a pixel
    pool_pintura, pool_foto = ImagePool(args.pool_size), ImagePool(args.pool_size)

    # As mesmas 8 fotos de teste em todas as épocas, para ver a evolução
    fixed_fotos = to_float(load_domain("foto", train=False, size=args.size)[:8]).to(device)

    logger = CSVLogger(out / "logs.csv",
                       ["epoch", "step", "lr", "loss_g_adv", "loss_f_adv", "loss_cycle",
                        "loss_identity", "loss_d_pintura", "loss_d_foto"],
                       append=args.resume is not None)

    step = (start_epoch - 1) * steps_per_epoch
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        G.train(); F.train()
        for i in range(steps_per_epoch):
            foto = random_batch(fotos, args.batch_size, device)
            pintura = random_batch(pinturas, args.batch_size, device)

            # --- Geradores
            opt_g.zero_grad(set_to_none=True)
            with amp():
                fake_pintura = G(foto)
                fake_foto = F(pintura)

                # Adversarial: enganar os discriminadores (alvo 1 = "real")
                pred = D_pintura(fake_pintura)
                loss_g_adv = mse(pred, torch.ones_like(pred))
                pred = D_foto(fake_foto)
                loss_f_adv = mse(pred, torch.ones_like(pred))

                # Cycle: ir e voltar tem que dar a imagem original
                loss_cycle = l1(F(fake_pintura), foto) + l1(G(fake_foto), pintura)

                # Identity: o que já está no domínio de chegada não deve mudar
                if args.lambda_identity > 0:
                    loss_identity = l1(G(pintura), pintura) + l1(F(foto), foto)
                else:
                    loss_identity = torch.zeros((), device=device)

                loss_g = (loss_g_adv + loss_f_adv
                          + args.lambda_cycle * loss_cycle
                          + args.lambda_identity * loss_identity)
            loss_g.backward()
            opt_g.step()

            # --- Discriminadores: real -> 1, falsa (do pool) -> 0
            opt_d.zero_grad(set_to_none=True)
            loss_d = {}
            for name, D, real, fake in (("pintura", D_pintura, pintura, pool_pintura.query(fake_pintura)),
                                        ("foto", D_foto, foto, pool_foto.query(fake_foto))):
                with amp():
                    pred_real, pred_fake = D(real), D(fake)
                    loss_d[name] = 0.5 * (mse(pred_real, torch.ones_like(pred_real))
                                          + mse(pred_fake, torch.zeros_like(pred_fake)))
            (loss_d["pintura"] + loss_d["foto"]).backward()
            opt_d.step()

            if step % args.log_every == 0:
                logger.log({
                    "epoch": epoch, "step": step, "lr": sched_g.get_last_lr()[0],
                    "loss_g_adv": loss_g_adv.item(), "loss_f_adv": loss_f_adv.item(),
                    "loss_cycle": loss_cycle.item(), "loss_identity": loss_identity.item(),
                    "loss_d_pintura": loss_d["pintura"].item(), "loss_d_foto": loss_d["foto"].item(),
                })
            if epoch == start_epoch and i == 99:
                per_step = (time.time() - t0) / 100
                print(f"  ~{per_step * 1000:.0f} ms por passo | "
                      f"estimativa: {per_step * steps_per_epoch / 60:.1f} min por época")
            step += 1

        sched_g.step()
        sched_d.step()

        # Grade: linha 1 = fotos, linha 2 = G(foto), linha 3 = F(G(foto)), a volta
        G.eval(); F.eval()
        with torch.no_grad():
            pint = G(fixed_fotos)
            grid = torch.cat([fixed_fotos, pint, F(pint)])
        save_grid(grid, out / "samples" / f"epoch_{epoch:03d}.png", nrow=8)

        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            save_checkpoint(out / "checkpoints" / f"epoch_{epoch:03d}.pt",
                            G=G.state_dict(), F=F.state_dict(),
                            D_pintura=D_pintura.state_dict(), D_foto=D_foto.state_dict(),
                            opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(),
                            sched_g=sched_g.state_dict(), sched_d=sched_d.state_dict(),
                            epoch=epoch, args=vars(args))
        print(f"Época {epoch:3d}/{args.epochs} | G_adv {loss_g_adv.item():.3f} "
              f"| cycle {loss_cycle.item():.3f} | D_pintura {loss_d['pintura'].item():.3f} "
              f"| {(time.time() - t0) / 60:.1f} min")

    make_gif(out / "samples", out / "evolucao.gif")
    print("Treino concluído.")


if __name__ == "__main__":
    main()
