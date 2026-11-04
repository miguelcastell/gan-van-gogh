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

## Feito: recorte aleatório (2026-10-30 a 2026-10-31)

Primeira mudança da seção 11 do notebook 03. Com ela, o modelo chegou a FID 117,4 (antes, 125,3). Detalhes no [registro do experimento](../experimentos/2026-10-30-recorte-aleatorio.md).

- [x] Treino de 40 épocas com `--load-size 143`, avaliação e gráficos
- [ ] Próximo treino: outra mudança da seção 11, mantendo o recorte e **escrevendo a previsão antes**

## Agora: pinceladas de verdade (combinado em 2026-10-31)

O objetivo do autor: que o gerador transforme **qualquer foto** numa pintura de Van Gogh, com pinceladas visíveis. O gerador já pinta a foto inteira (`src.stylize --sem-mascara`); a máscara do céu é só a composição final. Pincelada é detalhe fino, então os próximos treinos atacam a resolução e a capacidade do gerador, uma mudança por vez. As imagens de cada um ficam em [`data/treinos/`](../../data/treinos/README.md).

- [x] Treino 04: 256x256 ([registro](../experimentos/2026-10-31-resolucao-256.md)), concluído em 2026-11-01. Melhorou bastante a aparência; métricas quase iguais
- [x] Treino 05: 256x256 + 9 blocos residuais ([registro](../experimentos/2026-11-02-nove-blocos.md)), encerrado na época 30. Empatou com o treino 04
- [ ] ~~Treinar por mais passos~~: descartado por falta de tempo e de GPU (200 épocas em 256 levariam ~2 dias)
- [x] Treino 06: só fotos de céu marcado ([ADR 0008](../adr/0008-treinar-com-fotos-de-ceu.md), [registro](../experimentos/2026-11-03-fotos-de-ceu.md)), concluído em 2026-11-04. Sem ganho claro sobre o treino 04 (FID 127,1 contra 123,5)
- [ ] Comparar `cyclegan_256` e `cyclegan_256_ceu` com o `src.stylize` nas fotos da demonstração e escolher o modelo da entrega
- [x] Transferência de estilo neural com a Noite Estrelada, como comparação ([ADR 0009](../adr/0009-transferencia-neural-como-comparacao.md), [registro](../experimentos/2026-11-03-transferencia-neural.md))

**Descartado: filtrar só as pinturas com céu.** Reduziria as 400 pinturas para algo como 200, e o discriminador decoraria a coleção ainda mais rápido. Além disso, para pintar a foto inteira, as paisagens, os campos e as flores também ensinam pinceladas.

## Como melhorar as imagens (levantado em 2026-11-03)

O que os treinos 01 a 05 mostraram, do que mais muda o resultado para o que menos muda:

1. **O que o gerador vê no treino.** No pipeline só o céu é pintado, então treinar só com fotos de céu marcado concentra o gerador no que importa ([ADR 0008](../adr/0008-treinar-com-fotos-de-ceu.md), treino 06).
2. **A foto de entrada.** O modelo brilha em céus com estrutura (pôr do sol, nuvens marcadas) e falha em céu cinza liso, onde vira textura granulada. Na demonstração, use fotos boas e mostre um céu cinza como limitação.
3. **O `--size` no `src.stylize`.** Menor = pinceladas maiores e imagem mais borrada; maior = mais nítido, mas pinceladas pequenas. Teste 256, 384 e 512 em cada foto.
4. **Para a "Noite Estrelada no fundo", use a transferência neural** (`src.transferencia_estilo --so-ceu`, [ADR 0009](../adr/0009-transferencia-neural-como-comparacao.md)). A CycleGAN não cria redemoinhos onde a foto não tem estrutura; isso é limitação do método, não erro de treino.
5. **O que muda pouco:** mais capacidade (9 blocos, treino 05) e alguns treinos a mais iguais aos anteriores. Treinar 200 épocas como no artigo levaria ~2 dias de GPU.

**Cuidado com a RAM em treinos longos.** Em 2026-11-03, a primeira tentativa do treino 06 foi encerrada pelo Claude Code porque a RAM do sistema ficou baixa. A segunda, também pelo Claude Code, terminou sem problemas (6h20). Para garantir, feche o navegador antes ou rode num PowerShell próprio:

```powershell
.venv\Scripts\python.exe outputs\rodar_treino_06.py
```

## Entrega (até 2026-12-02)

Combinado em 2026-11-03: **apresentação em slides** e um **notebook de apresentação** com samples, enviado ao professor. O trabalho é em grupo, mas só o autor está fazendo.

- [x] Formato da entrega
- [x] Individual ou em grupo
- [ ] Tamanho esperado e critérios de avaliação
- [ ] Análise ética: riscos de uso, viés, privacidade e limites da geração sintética (o único item do enunciado que ainda falta)
- [ ] Notebook de apresentação: amostras, curvas, comparação entre treinos, CycleGAN x transferência neural, exemplos ruins
- [ ] Slides

Anterior: [Roteiro](02-roteiro-etapas.md) · Próximo: [Como aprender](04-como-aprender.md)
