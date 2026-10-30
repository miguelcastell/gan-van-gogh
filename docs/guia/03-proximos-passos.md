# 3. Próximos passos: o que fazer agora

Marque os itens conforme for concluindo.

## Feito: ambiente e dados

O venv tinha o PyTorch **só para CPU** (`2.15.0+cpu`). Ele foi trocado pela versão com CUDA (`2.15.0+cu132`), escolhida porque o driver da placa suporta CUDA até 13.3 (veja com `nvidia-smi`).

- [x] Venv criado e PyTorch com CUDA instalado
- [x] `python -c "import torch; print(torch.cuda.is_available())"` imprime `True`
- [x] Dataset baixado com `python -m src.download_data`. Ele vem de http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip e fica em `data/vangogh2photo/`

## Agora: entender os dados (notebook 01)

Para abrir os notebooks, com o venv ativado, na raiz do projeto:

```powershell
.venv\Scripts\activate
jupyter lab
```

Antes, leia o [guia 05 (KDD)](05-kdd.md).

- [ ] Completar os TODOs de `notebooks/01_exploracao.ipynb` e responder às perguntas
- [ ] Preencher a tabela de resumo no final e pedir para registrar em [`experimentos/`](../experimentos/)

## Depois: montar a CycleGAN (notebook 02)

Antes, leia o [guia 06 (a CycleGAN por dentro)](06-cyclegan-por-dentro.md).

- [ ] Completar os TODOs de `notebooks/02_cyclegan.ipynb`: bloco residual, gerador, discriminador e as três losses
- [ ] Rodar o treino curto (300 passos) com e sem *cycle loss*, **escrevendo a previsão antes**
- [ ] Registrar as previsões e o que observou em [`experimentos/`](../experimentos/)

## Depois: o treino longo, no terminal

O treino longo roda no terminal, não no notebook ([ADR 0006](../adr/0006-notebooks-e-scripts.md)).

```powershell
python -m src.train_cyclegan --epochs 1 --photos 300
```

- [ ] Conferir o tempo por passo com o dataset real. Com imagens falsas deu ~130 ms ([experimento](../experimentos/2026-10-29-tempo-por-passo.md))
- [ ] Decidir o número de épocas e de fotos ([ADR 0002](../adr/0002-dataset-e-modelos.md), seção "Em aberto")
- [ ] Rodar o treino completo e a versão `--sem-cycle`

## Esta semana: confirmar com o professor

- [ ] Formato da entrega (notebook, relatório, apresentação?)
- [ ] Tamanho esperado do texto e critérios de avaliação
- [ ] Individual ou em grupo

Anterior: [Roteiro](02-roteiro-etapas.md) · Próximo: [Como aprender](04-como-aprender.md)
