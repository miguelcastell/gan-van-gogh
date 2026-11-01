"""Galeria dos treinos: uma pasta por treino em data/treinos/, para ver a evolução.

Uso (na raiz do projeto):
    python -m src.galeria              # refaz a pasta de todos os treinos
    python -m src.galeria --cpu        # para não disputar a GPU com um treino em andamento

Para cada pasta de outputs/ com checkpoints (em ordem cronológica: treino_01, treino_02...):
    data/treinos/treino_NN_<nome>/
        grade.png   as 8 fotos fixas de teste no melhor checkpoint
                    (linhas: original, pintada, de volta a foto)
        ceu.png     o produto final: 4 fotos de teste com só o céu pintado, no melhor checkpoint
        README.md   parâmetros, métricas e link para o registro do experimento

"Melhor checkpoint" é o de menor FID entre os avaliados (os por_checkpoint.csv de outputs/
e o metrics.json). Se nenhum foi avaliado ainda, é o último salvo.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.comparar import treinos
from src.models import Generator
from src.stylize import pintar_ceu
from src.utils import DATASET_DIR, ROOT, get_device, last_checkpoint, load_image, to_unit

GALERIA = ROOT / "data" / "treinos"
DOCS_EXP = ROOT / "docs" / "experimentos"
# Fotos de testB com bastante céu: a 2ª, 5ª, 7ª e 8ª das grades por época.
FOTOS_CEU = ["2014-08-01 22_38_22.jpg", "2014-08-02 15_56_41.jpg",
             "2014-08-03 09_47_19.jpg", "2014-08-03 17_14_07.jpg"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="roda na CPU (para não disputar a GPU)")
    return p.parse_args()


def fids_por_epoca(nome: str) -> dict[int, dict]:
    """FID/KID de cada checkpoint avaliado deste treino."""
    linhas = {}
    for arq in (ROOT / "outputs").glob("*/por_checkpoint.csv"):
        with open(arq, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["modelo"] == nome:
                    linhas[int(r["epoca"])] = {"fid": float(r["fid"]), "kid": float(r["kid"])}
    return linhas


def registro_do_experimento(nome: str) -> Path | None:
    """O registro em docs/experimentos/ que cita a pasta outputs/<nome>/."""
    for doc in sorted(DOCS_EXP.glob("20*.md")):
        if f"outputs/{nome}/" in doc.read_text(encoding="utf-8"):
            return doc
    return None


def painel_ceu(G, size, device, destino: Path, titulo: str):
    fig, axes = plt.subplots(2, len(FOTOS_CEU), figsize=(3.2 * len(FOTOS_CEU), 6.8))
    for c, nome in enumerate(FOTOS_CEU):
        img = load_image(DATASET_DIR / "testB" / nome)
        x, _, _, resultado = pintar_ceu(G, img, size, device)
        for r, im in enumerate((x, resultado)):
            axes[r, c].imshow(to_unit(im[0]).clamp(0, 1).permute(1, 2, 0).cpu().numpy())
            axes[r, c].axis("off")
    for r, rotulo in enumerate(("original", "só o céu pintado")):
        axes[r, 0].text(-0.04, 0.5, rotulo, transform=axes[r, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=10)
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(destino, dpi=130)
    plt.close()


def readme(n, nome, ckpt_args, epoca, fid, metrics, registro, em_andamento):
    a = ckpt_args
    load = a.get("load_size") or a["size"]
    recorte = f"carrega em {load} e recorta {a['size']}" if load != a["size"] else "não (só espelhamento)"
    linhas = [
        f"# Treino {n:02d}: `{nome}`",
        "",
        "Gerado por `python -m src.galeria`; não edite à mão (o arquivo é refeito).",
        "",
    ]
    if em_andamento:
        linhas += ["**Treino em andamento:** as imagens são do último checkpoint salvo até agora.", ""]
    if registro:
        linhas += [f"Registro completo do experimento (pergunta, previsão, diário, análise): "
                   f"[`{registro.name}`](../../../docs/experimentos/{registro.name})", ""]
    linhas += [
        "## Imagens",
        "",
        f"Do checkpoint da **época {epoca}**"
        + (f", o de menor FID entre os avaliados ({fid:.1f})." if fid is not None else ", o último salvo."),
        "",
        "- [`grade.png`](grade.png): as 8 fotos fixas de teste. Linhas: original, pintada, de volta a foto.",
        "- [`ceu.png`](ceu.png): o produto final, com só o céu pintado (máscara do SegFormer).",
        "",
        "## Parâmetros",
        "",
        "| Parâmetro | Valor |",
        "|---|---|",
        f"| Resolução | {a['size']}x{a['size']} |",
        f"| Recorte aleatório | {recorte} |",
        f"| Épocas / fotos por época | {a['epochs']} / {a['photos']} |",
        f"| Blocos residuais no gerador | {a['n_blocks']} |",
        f"| *Cycle loss* | {'desligada' if a.get('sem_cycle') else 'ligada'} |",
        f"| λ_cycle / λ_identity | {a['lambda_cycle']} / {a['lambda_identity']} |",
        f"| lr / beta1 | {a['lr']} / {a['beta1']} |",
        f"| Queda do lr | a partir da época {a.get('decay_start') or a['epochs'] // 2} |",
        f"| Batch / image pool | {a['batch_size']} / {a['pool_size']} |",
        f"| Precisão mista (`--amp`) | {'sim' if a.get('amp') else 'não'} |",
        f"| Seed | {a['seed']} |",
        "",
    ]
    if metrics:
        linhas += [
            f"## Métricas (`{metrics['checkpoint']}`)",
            "",
            "| Métrica | Valor |",
            "|---|---|",
            f"| FID ↓ | {metrics['fid']:.1f} (fotos sem conversão: {metrics['fid_fotos_originais']:.1f}) |",
            f"| KID ↓ | {metrics['kid']:.4f} ± {metrics['kid_std']:.4f} |",
            f"| Diversidade | {metrics['diversidade']:.1f} (pinturas reais: {metrics['diversidade_pinturas']:.1f}) |",
            f"| Erro de ida e volta (`cycle_l1`) ↓ | {metrics['cycle_l1']:.3f} |",
            f"| Distância à pintura mais próxima | {metrics['nn_dist_fake_mean']:.1f} "
            f"(entre pinturas reais: {metrics['nn_dist_ref_mean']:.1f}) |",
            "",
            f"Em {a['size']}x{a['size']}: o FID só é comparável entre treinos da mesma resolução.",
            "",
        ]
    return "\n".join(linhas)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cpu") if args.cpu else get_device()
    GALERIA.mkdir(parents=True, exist_ok=True)
    indice = ["# Galeria dos treinos", "",
              "Uma pasta por treino, em ordem cronológica. Gerado por `python -m src.galeria`; "
              "a comparação lado a lado de todos está em "
              "[`report/figuras/comparacao_modelos.png`](../../report/figuras/comparacao_modelos.png).", "",
              "| Treino | Resolução | Melhor época | FID | Céu pintado |", "|---|---|---|---|---|"]

    for n, pasta in enumerate(treinos(), start=1):
        nome = pasta.name
        destino = GALERIA / f"treino_{n:02d}_{nome}"
        destino.mkdir(exist_ok=True)

        ultimo = last_checkpoint(pasta)
        metrics_path = pasta / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None

        fids = fids_por_epoca(nome)
        if metrics:
            fids.setdefault(int(metrics["checkpoint"][6:9]), {"fid": metrics["fid"]})
        disponiveis = {e: v for e, v in fids.items()
                       if (pasta / "checkpoints" / f"epoch_{e:03d}.pt").exists()}
        if disponiveis:
            epoca = min(disponiveis, key=lambda e: disponiveis[e]["fid"])
            fid = disponiveis[epoca]["fid"]
        else:
            epoca, fid = int(ultimo.stem[6:]), None
        ckpt = torch.load(pasta / "checkpoints" / f"epoch_{epoca:03d}.pt", map_location=device)
        a = ckpt["args"]
        em_andamento = not metrics or epoca > int(ultimo.stem[6:]) or int(ultimo.stem[6:]) < a["epochs"]

        shutil.copy(pasta / "samples" / f"epoch_{epoca:03d}.png", destino / "grade.png")
        G = Generator(n_blocks=a["n_blocks"]).to(device).eval()
        G.load_state_dict(ckpt["G"])
        painel_ceu(G, a["size"], device, destino / "ceu.png",
                   f"Treino {n:02d} ({nome}), época {epoca}")

        registro = registro_do_experimento(nome)
        (destino / "README.md").write_text(
            readme(n, nome, a, epoca, fid, metrics, registro, em_andamento), encoding="utf-8")
        pasta_rel = destino.name
        indice.append(f"| [{n:02d} `{nome}`]({pasta_rel}/README.md)"
                      + (" (em andamento)" if em_andamento else "")
                      + f" | {a['size']}px | {epoca} | {f'{fid:.1f}' if fid else '—'} "
                      f"| [ceu.png]({pasta_rel}/ceu.png) |")
        print(f"treino_{n:02d}_{nome}: época {epoca}" + (f", FID {fid:.1f}" if fid else ""))

    indice += ["", "O FID só é comparável entre treinos da mesma resolução.", ""]
    (GALERIA / "README.md").write_text("\n".join(indice), encoding="utf-8")


if __name__ == "__main__":
    main()
