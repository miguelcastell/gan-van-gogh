"""Gerador ResNet e Discriminador PatchGAN da CycleGAN (Zhu et al., 2017).

Explicação das peças: docs/guia/06-cyclegan-por-dentro.md.
"""
import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """Inicialização usada no artigo da CycleGAN: N(0, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ResnetBlock(nn.Module):
    """saída = entrada + correção. "Não mudar nada" é fácil: basta a correção ser zero."""

    def __init__(self, ch: int):
        super().__init__()
        self.correcao = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.correcao(x)


class Generator(nn.Module):
    """Imagem (N, 3, H, W) em [-1, 1] -> imagem (N, 3, H, W) em [-1, 1].

    É totalmente convolucional: treina em 128x128, mas aceita qualquer H e W múltiplos de 4.
    """

    def __init__(self, ngf: int = 64, n_blocks: int = 6):
        super().__init__()
        layers = [
            # Codificador: (3, 128, 128) -> (ngf, 128, 128)
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # -> (ngf*2, 64, 64)
            nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> (ngf*4, 32, 32)
            nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        ]
        # Transformação: os blocos residuais mantêm o tamanho (ngf*4, 32, 32)
        layers += [ResnetBlock(ngf * 4) for _ in range(n_blocks)]
        layers += [
            # Decodificador: -> (ngf*2, 64, 64)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> (ngf, 128, 128)
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            # -> (3, 128, 128)
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, 7),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """Imagem (N, 3, 128, 128) -> grade de notas (N, 1, 14, 14), sem sigmoid (LSGAN).

    Cada nota da grade julga um pedaço de ~70x70 pixels da imagem.
    """

    def __init__(self, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (3, 128, 128) -> (ndf, 64, 64). Sem norm na primeira camada, como no artigo.
            nn.Conv2d(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            # -> (ndf*2, 32, 32)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            # -> (ndf*4, 16, 16)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            # -> (ndf*8, 15, 15)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
            # -> (1, 14, 14): uma nota por pedaço
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
