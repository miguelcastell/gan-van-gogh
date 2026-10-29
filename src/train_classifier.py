"""Treina o classificador auxiliar usado para medir diversidade (cobertura de classes).

Uso: python -m src.train_classifier
"""
import torch
import torch.nn.functional as F

from src.models import Classifier
from src.utils import (OUTPUTS_DIR, get_device, load_fashion_mnist, set_seed)


def main(epochs: int = 8, batch_size: int = 256):
    set_seed(0)
    device = get_device()
    x_tr, y_tr = load_fashion_mnist(train=True, device=device)
    x_te, y_te = load_fashion_mnist(train=False, device=device)

    model = Classifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(x_tr.size(0), device=device)
        for i in range(0, x_tr.size(0), batch_size):
            b = idx[i:i + batch_size]
            loss = F.cross_entropy(model(x_tr[b]), y_tr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        print(f"Época {epoch} | acurácia teste {acc:.4f}")

    path = OUTPUTS_DIR / "classifier.pt"
    torch.save(model.state_dict(), path)
    print(f"Salvo em {path}")


if __name__ == "__main__":
    main()
