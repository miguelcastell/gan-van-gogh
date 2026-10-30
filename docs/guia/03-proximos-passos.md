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

## Feito: o primeiro treino longo (2026-10-29)

Rodou durante a noite: a CycleGAN completa e a versão sem *cycle loss*, 40 épocas de 2.000 fotos cada (~2h50 por treino), mais a avaliação. Tudo está no [registro do experimento](../experimentos/2026-10-29-primeiro-treino.md).

- [x] Tempo por passo com o dataset real: ~126 ms (4,2 min por época)
- [x] Treino completo e treino `--sem-cycle`
- [x] Avaliação e gráficos (`report/figuras/`, `report/tabela_comparativa.md`)

## Depois: entender o primeiro treino (notebook 03)

- [ ] Ler `notebooks/03_analise_primeiro_treino.ipynb` (de preferência depois dos notebooks 01 e 02) e responder às perguntas ✍️
- [ ] Testar o céu de Van Gogh numa foto sua: `python -m src.stylize sua_foto.jpg`
- [ ] Escolher a próxima mudança a testar (seção 11 do notebook 03) e **escrever a previsão** antes de treinar

## Esta semana: confirmar com o professor

- [ ] Formato da entrega (notebook, relatório, apresentação?)
- [ ] Tamanho esperado do texto e critérios de avaliação
- [ ] Individual ou em grupo

Anterior: [Roteiro](02-roteiro-etapas.md) · Próximo: [Como aprender](04-como-aprender.md)
