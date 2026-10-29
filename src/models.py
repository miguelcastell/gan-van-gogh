"""Gerador, Discriminador (DCGAN) e Crítico (WGAN-GP) para imagens 28x28."""
import torch
import torch.nn as nn

NZ = 100


def weights_init(m: nn.Module) -> None:
    """Inicialização recomendada no paper do DCGAN: N(0, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Generator(nn.Module):
    """z (N, 100) -> imagem (N, 1, 28, 28) em [-1, 1]."""

    def __init__(self, nz: int = NZ, ngf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # (nz, 1, 1) -> (ngf*4, 7, 7)
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # -> (ngf*2, 14, 14)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # -> (1, 28, 28)
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """Imagem -> logit (sem sigmoid; usar BCEWithLogitsLoss)."""

    def __init__(self, ndf: int = 64, batchnorm: bool = True):
        super().__init__()
        norm = nn.BatchNorm2d(ndf * 2) if batchnorm else nn.Identity()
        self.net = nn.Sequential(
            # (1, 28, 28) -> (ndf, 14, 14)
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (ndf*2, 7, 7)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            norm,
            nn.LeakyReLU(0.2, inplace=True),
            # -> (1, 1, 1)
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def Critic(ndf: int = 64) -> Discriminator:
    """Crítico da WGAN-GP: mesma arquitetura, sem BatchNorm (incompatível com o gradient penalty)."""
    return Discriminator(ndf=ndf, batchnorm=False)


class Classifier(nn.Module):
    """CNN simples para classificar Fashion-MNIST (usada na avaliação de diversidade)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
