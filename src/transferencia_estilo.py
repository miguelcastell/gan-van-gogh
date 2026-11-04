"""Transferência de estilo neural (Gatys, Ecker e Bethge, 2015), para comparar com a CycleGAN.

Não é uma GAN e não treina nenhuma rede. Uma VGG19 pré-treinada no ImageNet serve de
"olho", e o que é otimizado são os pixels da imagem:

- conteúdo: as ativações de uma camada profunda (conv4_2) devem ficar parecidas com as
  da foto. Camadas profundas guardam "o que está onde" (a estátua, o horizonte);
- estilo: as matrizes de Gram de várias camadas (conv1_1 a conv5_1) devem ficar parecidas
  com as da pintura. A Gram mede quais canais disparam juntos, sem olhar a posição: é
  uma descrição de textura (pinceladas, cores, redemoinhos) que ignora a composição.

Diferença para a CycleGAN: aqui o estilo vem de UM quadro (a Noite Estrelada), e cada
imagem nova exige uma otimização inteira (~1 min). A CycleGAN aprende o estilo geral de
400 pinturas e, depois de treinada, converte qualquer foto numa única passada.

Uso (na raiz do projeto):
    python -m src.transferencia_estilo foto.jpg --estilo data/estilos/noite_estrelada.jpg
    python -m src.transferencia_estilo foto.jpg --estilo ... --so-ceu   # pinta só o céu

Saída em outputs/transferencia/.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as nnf
from torchvision.models import VGG19_Weights, vgg19
from torchvision.utils import save_image

from src.sky import composite, sky_mask
from src.utils import OUTPUTS_DIR, get_device, load_image

# Índices das camadas em vgg19().features (depois da ReLU de cada conv citada)
CAMADAS_ESTILO = {"conv1_1": 1, "conv2_1": 6, "conv3_1": 11, "conv4_1": 20, "conv5_1": 29}
CAMADA_CONTEUDO = 22  # conv4_2
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESVIO = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("foto", type=Path)
    p.add_argument("--estilo", type=Path, required=True, help="imagem da pintura")
    p.add_argument("--max-side", type=int, default=768, help="maior lado da imagem otimizada")
    p.add_argument("--passos", type=int, default=300)
    p.add_argument("--peso-estilo", type=float, default=1e6)
    p.add_argument("--peso-conteudo", type=float, default=1.0)
    p.add_argument("--escala-estilo", type=float, default=1.0,
                   help="tamanho da pintura em relação à foto: maior = pinceladas maiores")
    p.add_argument("--so-ceu", action="store_true", help="cola o resultado só no céu")
    return p.parse_args()


class Extrator(torch.nn.Module):
    """VGG19 congelada que devolve as ativações das camadas de estilo e de conteúdo."""

    def __init__(self, device):
        super().__init__()
        ultima = max(*CAMADAS_ESTILO.values(), CAMADA_CONTEUDO)
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:ultima + 1].to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        # Max pooling troca de ativação aos saltos; o average pooling (como em Gatys)
        # deixa o gradiente mais suave e a textura mais limpa.
        for i, camada in enumerate(self.vgg):
            if isinstance(camada, torch.nn.MaxPool2d):
                self.vgg[i] = torch.nn.AvgPool2d(2, 2)
        self.media, self.desvio = MEDIA.to(device), DESVIO.to(device)

    def forward(self, x):  # x em [0, 1]
        x = (x - self.media) / self.desvio
        estilo, conteudo = [], None
        for i, camada in enumerate(self.vgg):
            x = camada(x)
            if i in CAMADAS_ESTILO.values():
                estilo.append(x)
            if i == CAMADA_CONTEUDO:
                conteudo = x
        return estilo, conteudo


def gram(f: torch.Tensor) -> torch.Tensor:
    b, c, h, w = f.shape
    f = f.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


def redimensionar(img: torch.Tensor, max_side: int) -> torch.Tensor:
    h, w = img.shape[-2:]
    s = max_side / max(h, w)
    return nnf.interpolate(img, size=(round(h * s), round(w * s)), mode="bicubic",
                           align_corners=False).clamp(0, 1)


def transferir(extrator, foto, pintura, passos, peso_estilo, peso_conteudo):
    """foto e pintura em [0, 1], (1, 3, H, W). Devolve a imagem otimizada e o histórico das losses."""
    with torch.no_grad():
        alvo_estilo = [gram(f) for f in extrator(pintura)[0]]
        alvo_conteudo = extrator(foto)[1]
    img = foto.clone().requires_grad_(True)  # começa da foto: converge mais rápido que do ruído
    opt = torch.optim.LBFGS([img], max_iter=passos, history_size=50, line_search_fn="strong_wolfe")
    hist = []

    def closure():
        opt.zero_grad()
        estilo, conteudo = extrator(img.clamp(0, 1))
        l_estilo = sum(nnf.mse_loss(gram(f), g) for f, g in zip(estilo, alvo_estilo))
        l_conteudo = nnf.mse_loss(conteudo, alvo_conteudo)
        loss = peso_estilo * l_estilo + peso_conteudo * l_conteudo
        loss.backward()
        hist.append((l_estilo.item(), l_conteudo.item()))
        return loss

    opt.step(closure)
    return img.detach().clamp(0, 1), hist


def main():
    args = parse_args()
    device = get_device()
    foto = redimensionar(load_image(args.foto)[None].float().div(255), args.max_side).to(device)
    pintura = load_image(args.estilo)[None].float().div(255)
    pintura = redimensionar(pintura, round(args.max_side * args.escala_estilo)).to(device)

    extrator = Extrator(device)
    resultado, hist = transferir(extrator, foto, pintura, args.passos, args.peso_estilo,
                                 args.peso_conteudo)
    print(f"{len(hist)} avaliações | loss de estilo {hist[0][0]:.2e} -> {hist[-1][0]:.2e} | "
          f"conteúdo {hist[0][1]:.2f} -> {hist[-1][1]:.2f}")

    if args.so_ceu:
        foto_u8 = (foto[0] * 255).round().byte()
        mask = sky_mask(foto_u8, device)[None]
        resultado = composite(foto * 2 - 1, resultado * 2 - 1, mask).add(1).div(2)

    out = OUTPUTS_DIR / "transferencia"
    out.mkdir(parents=True, exist_ok=True)
    nome = f"{args.foto.stem}_{args.estilo.stem}{'_ceu' if args.so_ceu else ''}.png"
    save_image(resultado, out / nome)
    print(f"Salvo em {out / nome}")


if __name__ == "__main__":
    main()
