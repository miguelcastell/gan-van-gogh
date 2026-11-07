"""Treinos 07 e 08: os dois continuam o treino 06 da época 40 até a 80.

07 `cyclegan_256_ceu_80`: mesma receita do 06, só mais épocas (o FID ainda caía na época 40).
08 `cyclegan_256_ceu_id1`: igual ao 07, mas com λ_identity = 1 (era 5), para o gerador transformar mais.

LR nas épocas 41-80: --decay-start 0 com --epochs 80, ou seja, 1e-4 na época 41 caindo em linha
reta até 0 na 80 (fator 1 - época/80). Para o treino não começar com o LR de volta no máximo.

Em sequência: treino 07, avaliação dos checkpoints do 07, treino 08, avaliação do 08, e no fim
plots, comparar e galeria. Nada roda em paralelo na GPU.
"""
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\Miguel\Desktop\codiguin\gan")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
BASE = "cyclegan_256_ceu"
RESUME = f"outputs/{BASE}/checkpoints/epoch_040.pt"
LOGS = ROOT / "outputs" / "treino_07_08"
LOGS.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
COMUM = ["src.train_cyclegan", "--epochs", "80", "--decay-start", "0", "--size", "256",
         "--load-size", "286", "--n-blocks", "6", "--photos-list", "data/selecao_ceu/selecionadas.txt",
         "--resume", RESUME]
TREINOS = [
    ("cyclegan_256_ceu_80", ["--lambda-identity", "5"]),
    ("cyclegan_256_ceu_id1", ["--lambda-identity", "1"]),
]

resumo = {"inicio": datetime.now().isoformat(timespec="seconds"), "passos": []}


def salvar():
    (LOGS / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


def passo(nome, args):
    ini = time.time()
    reg = {"passo": nome, "inicio": datetime.now().isoformat(timespec="seconds")}
    with open(LOGS / f"{nome}.log", "w", encoding="utf-8") as log:
        cod = subprocess.call([PY, "-m", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    reg.update(minutos=round((time.time() - ini) / 60, 1), codigo_saida=cod)
    resumo["passos"].append(reg)
    salvar()
    return cod


def avaliar(modelo):
    ckpts = sorted((ROOT / "outputs" / modelo / "checkpoints").glob("epoch_*.pt"))
    linhas = []
    for ck in ckpts:
        if passo(f"avaliacao_{modelo}_{ck.stem}", ["src.evaluate", "--model", modelo, "--checkpoint", ck.name]) == 0:
            m = json.loads((ROOT / "outputs" / modelo / "metrics.json").read_text(encoding="utf-8"))
            linhas.append([modelo, int(ck.stem.split("_")[1]), f"{m['fid']:.1f}", f"{m['kid']:.4f}",
                           f"{m['diversidade']:.1f}", f"{m['cycle_l1']:.3f}"])
            with open(ROOT / "outputs" / modelo / "por_checkpoint.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["modelo", "epoca", "fid", "kid", "diversidade", "cycle_l1"])
                w.writerows(linhas)
    if ckpts:
        passo(f"avaliacao_final_{modelo}", ["src.evaluate", "--model", modelo])  # metrics.json = época 80


gpu = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu",
     "--format=csv", "-l", "30", "-f", str(LOGS / "gpu.csv")])
try:
    for modelo, extra in TREINOS:
        pasta = ROOT / "outputs" / modelo
        if (pasta / "checkpoints" / "epoch_080.pt").exists():
            continue  # já treinado e avaliado (o 07 terminou antes do encerramento por falta de RAM)
        pasta.mkdir(parents=True, exist_ok=True)
        # As curvas das épocas 1-40 são as do treino 06; com --resume o logs.csv continua daqui
        if not (pasta / "logs.csv").exists():
            shutil.copy(ROOT / "outputs" / BASE / "logs.csv", pasta / "logs.csv")
        if passo(f"treino_{modelo}", [*COMUM, *extra, "--out", modelo]) == 0:
            avaliar(modelo)
    passo("plots", ["src.plots"])
    passo("comparar", ["src.comparar"])
    passo("galeria", ["src.galeria"])
finally:
    gpu.terminate()
    resumo["fim"] = datetime.now().isoformat(timespec="seconds")
    salvar()
