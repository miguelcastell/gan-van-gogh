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


def plot_curves(name, df, window=20):
    """Com batch 1, cada loss oscila muito de um passo para o outro.
    A média móvel (`window` registros) deixa ver a tendência."""
    smooth = df.rolling(window, min_periods=1).mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for col, label in (("loss_g_adv", "G adversarial (foto → pintura)"),
                       ("loss_f_adv", "F adversarial (pintura → foto)"),
                       ("loss_cycle", "cycle"), ("loss_identity", "identity")):
        axes[0].plot(df.step, smooth[col], label=label, lw=1)
    axes[0].set_title(f"{name} — geradores")
    for col, label in (("loss_d_pintura", "D_pintura"), ("loss_d_foto", "D_foto")):
        axes[1].plot(df.step, smooth[col], label=label, lw=1)
    axes[1].axhline(0.25, ls="--", c="gray", label="D chutando (0.25)")
    axes[1].set_title(f"{name} — discriminadores")
    for ax in axes:
        ax.set_xlabel("passo")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{name}_curvas.png", dpi=150)
    plt.close()


def comparison_table():
    rows = []
    for path in sorted(OUTPUTS_DIR.glob("*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"modelo": m["model"], "λ_cycle": m["lambda_cycle"],
                     "FID": round(m["fid"], 1),
                     "FID fotos originais": round(m["fid_fotos_originais"], 1),
                     "KID": f"{m['kid']:.4f} ± {m['kid_std']:.4f}",
                     "diversidade": round(m["diversidade"], 1),
                     "diversidade pinturas": round(m["diversidade_pinturas"], 1),
                     "erro ida e volta": round(m["cycle_l1"], 3),
                     "dist. NN saída": round(m["nn_dist_fake_mean"], 1),
                     "dist. NN ref.": round(m["nn_dist_ref_mean"], 1)})
    if rows:
        table = pd.DataFrame(rows).to_markdown(index=False)
        (ROOT / "report" / "tabela_comparativa.md").write_text(table, encoding="utf-8")
        print(table)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for logs in sorted(OUTPUTS_DIR.glob("*/logs.csv")):
        plot_curves(logs.parent.name, pd.read_csv(logs))
    comparison_table()
    print(f"Figuras em {FIG_DIR}")


if __name__ == "__main__":
    main()
