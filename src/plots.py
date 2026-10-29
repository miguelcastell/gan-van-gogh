"""Curvas de treino e tabela comparativa a partir de outputs/*/logs.csv e metrics.json.

Uso: python -m src.plots
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import OUTPUTS_DIR, ROOT

FIG_DIR = ROOT / "report" / "figuras"


def plot_dcgan(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(df.step, df.loss_d, label="loss D", lw=0.8)
    axes[0].plot(df.step, df.loss_g, label="loss G", lw=0.8)
    axes[0].set_title("DCGAN — losses")
    axes[1].plot(df.step, df.d_x, label="D(x)", lw=0.8)
    axes[1].plot(df.step, df.d_g_z, label="D(G(z))", lw=0.8)
    axes[1].axhline(0.5, ls="--", c="gray")
    axes[1].set_title("DCGAN — saídas do discriminador")
    for ax in axes:
        ax.set_xlabel("passo")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dcgan_curvas.png", dpi=150)
    plt.close()


def plot_wgan(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].plot(df.step, df.loss_c, label="loss crítico", lw=0.8)
    axes[0].plot(df.step, df.loss_g, label="loss G", lw=0.8)
    axes[0].set_title("WGAN-GP — losses")
    axes[1].plot(df.step, df.wasserstein, label="estimativa Wasserstein", lw=0.8)
    axes[1].set_title("WGAN-GP — distância estimada")
    for ax in axes:
        ax.set_xlabel("passo do gerador")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wgan_gp_curvas.png", dpi=150)
    plt.close()


def comparison_table():
    rows = []
    for name in ("dcgan", "wgan_gp"):
        path = OUTPUTS_DIR / name / "metrics.json"
        if path.exists():
            m = json.loads(path.read_text())
            rows.append({"modelo": name, "FID": round(m["fid"], 2),
                         "entropia classes": round(m["class_entropy_norm"], 3),
                         "confiança média": round(m["mean_confidence"], 3),
                         "dist. NN gerado": round(m["nn_dist_fake_mean"], 2),
                         "dist. NN teste (ref.)": round(m["nn_dist_test_mean"], 2)})
    if rows:
        table = pd.DataFrame(rows).to_markdown(index=False)
        (ROOT / "report" / "tabela_comparativa.md").write_text(table, encoding="utf-8")
        print(table)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if (OUTPUTS_DIR / "dcgan" / "logs.csv").exists():
        plot_dcgan(pd.read_csv(OUTPUTS_DIR / "dcgan" / "logs.csv"))
    if (OUTPUTS_DIR / "wgan_gp" / "logs.csv").exists():
        plot_wgan(pd.read_csv(OUTPUTS_DIR / "wgan_gp" / "logs.csv"))
    comparison_table()
    print(f"Figuras em {FIG_DIR}")


if __name__ == "__main__":
    main()
