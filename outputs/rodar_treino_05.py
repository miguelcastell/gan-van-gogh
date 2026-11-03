"""Roda o treino 05 inteiro: treino, avaliação de cada checkpoint, gráficos, comparação e galeria."""
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\Miguel\Desktop\codiguin\gan")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
MODELO = "cyclegan_256_9blocos"
LOGS = ROOT / "outputs" / "treino_256_9blocos"
LOGS.mkdir(parents=True, exist_ok=True)
RESUME = sys.argv[1] if len(sys.argv) > 1 else None  # checkpoint para retomar
SUF = "_retomada" if RESUME else ""
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

resumo = {"inicio": datetime.now().isoformat(timespec="seconds"), "passos": []}


def salvar():
    (LOGS / f"resumo{SUF}.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


def passo(nome, args):
    ini = time.time()
    reg = {"passo": nome, "inicio": datetime.now().isoformat(timespec="seconds")}
    with open(LOGS / f"{nome}.log", "w", encoding="utf-8") as log:
        cod = subprocess.call([PY, "-m", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    reg.update(minutos=round((time.time() - ini) / 60, 1), codigo_saida=cod)
    resumo["passos"].append(reg)
    salvar()
    return cod


gpu = subprocess.Popen(
    ["nvidia-smi", "--query-gpu=timestamp,temperature.gpu,power.draw,memory.used,utilization.gpu",
     "--format=csv", "-l", "30", "-f", str(LOGS / f"gpu{SUF}.csv")])
try:
    cod = passo(f"treino{SUF}", ["src.train_cyclegan", "--epochs", "40", "--photos", "2000", "--size", "256",
                           "--load-size", "286", "--n-blocks", "9", "--out", MODELO]
               + (["--resume", RESUME] if RESUME else []))
    ckpts = sorted((ROOT / "outputs" / MODELO / "checkpoints").glob("epoch_*.pt"))
    linhas = []
    for ck in ckpts:
        if passo(f"avaliacao_{ck.stem}", ["src.evaluate", "--model", MODELO, "--checkpoint", ck.name]) == 0:
            m = json.loads((ROOT / "outputs" / MODELO / "metrics.json").read_text(encoding="utf-8"))
            linhas.append([MODELO, int(ck.stem.split("_")[1]), f"{m['fid']:.1f}", f"{m['kid']:.4f}",
                           f"{m['diversidade']:.1f}", f"{m['cycle_l1']:.3f}"])
            with open(LOGS / "por_checkpoint.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["modelo", "epoca", "fid", "kid", "diversidade", "cycle_l1"])
                w.writerows(linhas)
    if ckpts:
        passo("avaliacao_final", ["src.evaluate", "--model", MODELO])  # metrics.json = último checkpoint
        passo("plots", ["src.plots"])
        passo("comparar", ["src.comparar"])
        passo("galeria", ["src.galeria"])
finally:
    gpu.terminate()
    resumo["fim"] = datetime.now().isoformat(timespec="seconds")
    salvar()
