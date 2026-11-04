"""Separa as fotos de treino com céu "interessante" (pôr do sol, nuvens marcadas).

No fim do pipeline só o céu é pintado, então o gerador não precisa aprender a pintar
florestas ou cidades. Este script mede, em cada foto de trainB:

- fração de céu: média da probabilidade de "céu" do segmentador (o mesmo do stylize);
- colorido do céu: métrica de Hasler e Süsstrunk (2003), alta em pôr do sol;
- variação de luz no céu: desvio padrão da luminância, alto quando há nuvens marcadas.

Fica com as fotos que têm pelo menos `--min-ceu` de céu e ordena pela soma das duas
medidas padronizadas (z-score). Céu cinza liso e céu azul chapado ficam no fim da fila.

Uso (na raiz do projeto):
    python -m src.selecionar_fotos --n 1500

Saída: data/selecao_ceu/ranking.csv (todas as fotos), selecionadas.txt (os índices
usados pelo treino com --photos-list) e amostra.png (as 64 primeiras e 32 cortadas na borda).
"""
import argparse
import csv

import torch
import torch.nn.functional as nnf
from torchvision.utils import save_image

from src.sky import _load_segmenter
from src.utils import DATA_DIR, DATASET_DIR, get_device, load_domain


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500, help="quantas fotos separar")
    p.add_argument("--min-ceu", type=float, default=0.25, help="fração mínima de céu na foto")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def medir(fotos: torch.Tensor, device, batch_size: int):
    """Para cada foto uint8 (N, 3, H, W): fração de céu, colorido e desvio da luminância no céu."""
    processor, model, sky_id = _load_segmenter(device)
    medidas = []
    for i in range(0, fotos.size(0), batch_size):
        x = fotos[i:i + batch_size]
        inputs = processor(images=list(x.permute(0, 2, 3, 1).numpy()), return_tensors="pt").to(device)
        logits = nnf.interpolate(model(**inputs).logits, size=x.shape[2:], mode="bilinear",
                                 align_corners=False)
        ceu = logits.softmax(1)[:, sky_id]                       # (B, H, W), probabilidade
        rgb = x.to(device).float() / 255
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        rg, yb = r - g, 0.5 * (r + g) - b                        # eixos de cor de Hasler e Süsstrunk
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        for j in range(x.size(0)):
            m = ceu[j] > 0.5
            frac = ceu[j].mean().item()
            if m.sum() < 100:                                    # quase sem céu: nada a medir
                medidas.append((frac, 0.0, 0.0))
                continue
            a, c = rg[j][m], yb[j][m]
            colorido = (torch.sqrt(a.std() ** 2 + c.std() ** 2)
                        + 0.3 * torch.sqrt(a.mean() ** 2 + c.mean() ** 2)).item()
            medidas.append((frac, colorido, lum[j][m].std().item()))
        print(f"\r{min(i + batch_size, fotos.size(0))}/{fotos.size(0)}", end="", flush=True)
    print()
    return torch.tensor(medidas)


def main():
    args = parse_args()
    device = get_device()
    fotos = load_domain("foto", train=True, size=256)
    nomes = sorted(p.name for p in (DATASET_DIR / "trainB").glob("*.jpg"))
    assert len(nomes) == fotos.size(0), "o cache de fotos não bate com a pasta trainB"

    med = medir(fotos, device, args.batch_size)
    frac, colorido, var_luz = med.unbind(1)
    ok = frac >= args.min_ceu
    z = lambda v: (v - v[ok].mean()) / v[ok].std()
    nota = torch.where(ok, z(colorido) + z(var_luz), torch.tensor(float("-inf")))
    ordem = nota.argsort(descending=True)
    n = min(args.n, int(ok.sum()))
    escolhidas = ordem[:n]

    out = DATA_DIR / "selecao_ceu"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["posicao", "indice", "arquivo", "fracao_ceu", "colorido", "variacao_luz", "nota",
                    "selecionada"])
        for pos, i in enumerate(ordem.tolist()):
            w.writerow([pos + 1, i, nomes[i], f"{frac[i]:.3f}", f"{colorido[i]:.4f}",
                        f"{var_luz[i]:.4f}", f"{nota[i]:.3f}", int(pos < n)])
    (out / "selecionadas.txt").write_text("\n".join(map(str, sorted(escolhidas.tolist()))) + "\n")

    # Conferência visual: as 64 melhores e as 32 últimas que ainda entraram
    amostra = torch.cat([fotos[ordem[:64]], fotos[ordem[n - 32:n]]]).float() / 255
    save_image(nnf.interpolate(amostra, size=128), out / "amostra.png", nrow=16, padding=2)
    print(f"{int(ok.sum())} fotos com pelo menos {args.min_ceu:.0%} de céu; separadas: {n}")
    print(f"Salvo em {out}")


if __name__ == "__main__":
    main()
