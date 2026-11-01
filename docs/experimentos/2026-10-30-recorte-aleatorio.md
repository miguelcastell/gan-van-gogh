# Experimento: recorte aleatório no aumento de dados

- **Data:** 2026-10-30 (épocas 1 a 20, início às 20h04) e 2026-10-31 (épocas 21 a 40, início às 13h15)
- **Commit do código:** `3e5f8a2` (o código do treino não mudou na retomada)
- **Comandos:**
  ```powershell
  # 1ª parte (interrompida de propósito depois da época 20)
  python -m src.train_cyclegan --epochs 40 --photos 2000 --load-size 143 --out cyclegan_crop
  # 2ª parte (retomada)
  python -m src.train_cyclegan --epochs 40 --photos 2000 --load-size 143 --out cyclegan_crop --resume outputs/cyclegan_crop/checkpoints/epoch_020.pt
  python -m src.evaluate --model cyclegan_crop
  python -m src.plots
  # métricas de cada checkpoint (epoch_005.pt ... epoch_040.pt), terminando no 040
  python -m src.evaluate --model cyclegan_crop --checkpoint epoch_0XX.pt
  ```
- **Pasta de saída:** `outputs/cyclegan_crop/`. Logs brutos em `outputs/treino_crop/`: os arquivos sem sufixo são da 1ª parte, os `*_retomada` da 2ª. Métricas por checkpoint em [`por_checkpoint.csv`](../../outputs/treino_crop/por_checkpoint.csv).

## Pergunta

No [primeiro treino](2026-10-29-primeiro-treino.md), o D_pintura terminou muito à frente do gerador (loss ~0,03, contra 0,25 de quem chuta), e o FID parou de melhorar por volta da época 25. A suspeita é que ele **decorou as 400 pinturas**, que só podiam chegar de 2 jeitos: normais ou espelhadas.

Se cada pintura chegar recortada numa posição sorteada, o D_pintura deixa de decorar e o resultado melhora?

## Previsão (antes de rodar)

Nas palavras do autor, escritas com o treino já iniciado, mas antes de ver qualquer resultado:

> "Acho que vai ficar melhor esse treino, espero que o data augmentation melhore."

Para conferir no fim, "melhor" em relação à CycleGAN completa do primeiro treino (época 40) pode ser medido assim:
- FID abaixo de 125,3 e KID abaixo de 0,051;
- D_pintura terminando acima de ~0,03, ou seja, menos dominante;
- diversidade perto da das pinturas reais (137,7), sem colapso.

## Configuração

Igual à CycleGAN completa do primeiro treino, mudando só o recorte. Assim, qualquer diferença vem dele.

| Parâmetro | Valor |
|---|---|
| Épocas / fotos por época | 40 / 2.000 (80.000 passos) |
| λ_cycle / λ_identity | 10 / 5 |
| Queda do lr | a partir da época 20 |
| Seed | 42 |
| **Aumento de dados** | **carrega em 143x143 e recorta 128x128 numa posição sorteada** (`--load-size 143`), mais o espelhamento de antes. No primeiro treino, só o espelhamento |

Com o recorte, cada pintura pode chegar de 16 × 16 × 2 = 512 jeitos, contra 2 antes. Exemplos em `outputs/exemplos_aumento/` (fora do git).

## Dados

- **Treino:** as 400 pinturas de `trainA` e 2.000 fotos de `trainB` sorteadas com a seed 42, as mesmas do primeiro treino. As duas recebem o recorte. As imagens vêm do cache `data/vangogh2photo_143/`.
- **Grades por época:** as 8 primeiras fotos de `testB`, inteiras, em 128x128, sem recorte.
- **Avaliação:** as 751 fotos de `testB` contra as 400 pinturas, sem recorte. É o mesmo método do primeiro treino, então os números são comparáveis.

## Diário da execução

| Hora | Evento |
|---|---|
| 20:04 | Início. Primeiros 100 passos: ~871 ms por passo (estimativa de 29 min por época). Nesses 2 primeiros minutos, a GPU estava limitada a ~25 W (motivo `0x24` no `gpu.csv`, limite de energia). Às 20:06, voltou a 71 W e ~1.927 MHz, sem limite, como no primeiro treino. A medida inicial está contaminada por esse trecho |
| 20:27 | Época 5 (1º checkpoint). A partir da época 2, **4,3 min por época**, o mesmo ritmo do primeiro treino: o recorte não deixou o treino mais lento |
| 21:13 | Época 15. Média do D_pintura por época: 0,176 (época 5), 0,211 (10) e 0,213 (15), contra 0,186, 0,151 e 0,120 no primeiro treino. **No primeiro treino ele caía sem parar; com o recorte, estabilizou perto de 0,21**, mais perto do 0,25 de quem chuta |
| 21:34 | Época 20 (4º checkpoint). **Treino interrompido de propósito**, a pedido do autor, logo depois de salvar o `epoch_020.pt`. A avaliação e os gráficos rodaram em seguida, sobre esse checkpoint |
| 31/10 13:15 | **Retomada** a partir do `epoch_020.pt`, com as mesmas opções (`--load-size 143` incluso). O log confirma "Continuando da época 21" |
| 31/10 14:53 | Época 40. 20 épocas em 98,2 min: 4,4 a 4,7 min por época até a época 30 e 5,0 a 5,3 min da 31 em diante. O `gpu_retomada.csv` não mostra limite de energia nem queda de clock nesse trecho (~1.900 MHz, ~65 W, 70–72 °C). **A causa da lentidão não foi identificada** |
| 31/10 14:54 | Avaliação e gráficos sobre o `epoch_040.pt` (~1 min, [`resumo_retomada.json`](../../outputs/treino_crop/resumo_retomada.json)), todos com código de saída 0 |

## Resultado (parcial: época 20 de 40)

**Tempo:** 20 épocas em 92 min (4,3 a 4,6 min por época). Avaliação e gráficos: ~1 min ([`resumo.json`](../../outputs/treino_crop/resumo.json)). O `codigo_saida` 1 do treino é a interrupção manual, não um erro.

**Métricas** (751 fotos de teste, contra as 400 pinturas; [`metrics.json`](../../outputs/cyclegan_crop/metrics.json)). A comparação justa é com a **mesma época 20** do primeiro treino ([`por_checkpoint.csv`](../../outputs/primeiro_treino/por_checkpoint.csv)); a época 40 entra como referência do melhor resultado anterior:

| | Com recorte, época 20 | 1º treino, época 20 | 1º treino, época 40 | Referência |
|---|---|---|---|---|
| FID ↓ | **126,0** | 136,9 | 125,3 | 179,8 (fotos sem conversão) |
| KID ↓ | **0,042 ± 0,005** | 0,056 | 0,051 | 0,100 (fotos sem conversão) |
| Diversidade | 135,3 | 137,8 | 128,9 | 137,7 (pinturas reais) |
| Erro de ida e volta (`cycle_l1`) ↓ | 0,147 | 0,141 | 0,130 | — |
| Mudança foto → pintura (`mudanca_l1`) | 0,231 | não medido | 0,295 | — |
| Distância à pintura mais próxima (média / mínima) | 96,0 / 63,5 | não medido | 93,1 / 58,2 | 93,2 (entre pinturas reais) |

**Losses, média por época** ([`logs.csv`](../../outputs/cyclegan_crop/logs.csv)):

| Época | D_pintura com recorte | D_pintura 1º treino | G_adv com recorte | G_adv 1º treino |
|---|---|---|---|---|
| 5 | 0,176 | 0,186 | 0,426 | 0,512 |
| 10 | 0,211 | 0,151 | 0,455 | 0,513 |
| 15 | 0,213 | 0,120 | 0,461 | 0,591 |
| 20 | 0,167 | 0,103 | 0,456 | 0,535 |

**Figuras:** [grade da época 20](../../outputs/cyclegan_crop/samples/epoch_020.png) (compare com a [do 1º treino](../../outputs/cyclegan/samples/epoch_020.png)), [curvas](../../report/figuras/cyclegan_crop_curvas.png), [piores casos](../../report/figuras/cyclegan_crop_piores.png), [vizinhos](../../report/figuras/cyclegan_crop_vizinhos.png), [memorização](../../report/figuras/cyclegan_crop_memorizacao.png).

## O que observei (até a época 20)

1. **O D_pintura ficou menos dominante.** Na época 20, a média dele foi 0,167, contra 0,103 no primeiro treino, e o G_adv ficou mais baixo em todas as épocas medidas. O jogo está mais equilibrado, que era o objetivo do recorte. Isso reforça a hipótese de que, no primeiro treino, ele estava decorando as pinturas, mas não a prova.
2. **Com metade do treino, já empatou com o melhor resultado anterior.** FID 126,0 na época 20, contra 136,9 na mesma época do primeiro treino e 125,3 no fim dele. O KID (0,042) já é o melhor do projeto.
3. **A diversidade ficou mais perto das pinturas reais** (135,3 contra 137,7), sem sinal de colapso.
4. **A preservação do conteúdo ficou quase igual** (`cycle_l1` 0,147 contra 0,141 na mesma época). O recorte não atrapalhou a volta.
5. **A pintura muda menos a foto** (`mudanca_l1` 0,231 contra 0,295 da época 40 do 1º treino), mas parte disso pode ser só a época mais cedo, já que no 1º treino essa medida não foi tirada na época 20.
6. **Nada de cópia:** as saídas ficaram até um pouco mais longe das pinturas reais (96,0) que as pinturas entre si (93,2).
7. **Na grade da época 20, o céu liso da casinha (5ª foto) não tem mais o padrão de grade e as listras verticais** do primeiro treino: virou textura de pincelada. O pato branco aparece (amarelado) em vez de sumir, e o templo está menos verde. É uma grade só, de 8 fotos; os números acima são o que vale.

**Sobre a previsão do autor ("vai ficar melhor"):** até aqui, confirmada na comparação justa (mesma época). Contra os critérios escritos, que comparam com a época 40 do primeiro treino: KID abaixo de 0,051 ✓, D_pintura acima de 0,03 ✓, diversidade perto de 137,7 ✓, FID abaixo de 125,3 ainda não (126,0, praticamente empatado). A conclusão final depende das épocas 21 a 40.

## Resultado final (época 40)

**Tempo total:** 40 épocas em 190 min (92 + 98,2), contra ~170 min do primeiro treino.

**Métricas** (751 fotos de teste, contra as 400 pinturas; [`metrics.json`](../../outputs/cyclegan_crop/metrics.json)):

| | Com recorte, época 40 | 1º treino, época 40 | Referência |
|---|---|---|---|
| FID ↓ | **117,4** | 125,3 | 179,8 (fotos sem conversão) |
| KID ↓ | **0,036 ± 0,004** | 0,051 ± 0,005 | 0,100 (fotos sem conversão) |
| Diversidade | 130,8 | 128,9 | 137,7 (pinturas reais) |
| Erro de ida e volta (`cycle_l1`) ↓ | **0,126** | 0,130 | — |
| Mudança foto → pintura (`mudanca_l1`) | 0,260 | 0,295 | — |
| Distância à pintura mais próxima (média / mínima) | 93,1 / 57,6 | 93,1 / 58,2 | 93,2 (entre pinturas reais) |

**Métricas por checkpoint** ([`por_checkpoint.csv`](../../outputs/treino_crop/por_checkpoint.csv); o 1º treino vem de [`primeiro_treino/por_checkpoint.csv`](../../outputs/primeiro_treino/por_checkpoint.csv)). A avaliação é determinística: reavaliar as épocas 20 e 40 deu exatamente os mesmos números de antes.

| Época | FID com recorte | FID 1º treino | KID com recorte | KID 1º treino |
|---|---|---|---|---|
| 5 | 178,0 | 168,2 | 0,103 | 0,087 |
| 10 | 153,6 | 137,8 | 0,077 | 0,053 |
| 15 | 141,7 | 144,5 | 0,058 | 0,065 |
| 20 | 126,0 | 136,9 | 0,042 | 0,056 |
| 25 | 120,4 | 124,2 | 0,037 | 0,046 |
| 30 | 122,4 | 130,9 | 0,038 | 0,057 |
| 35 | 122,1 | 128,2 | 0,040 | 0,054 |
| 40 | **117,4** | 125,3 | **0,036** | 0,052 |

**Losses, média por época** (calculadas de [`logs.csv`](../../outputs/cyclegan_crop/logs.csv)):

| Época | D_pintura com recorte | D_pintura 1º treino | G_adv com recorte | G_adv 1º treino |
|---|---|---|---|---|
| 20 | 0,167 | 0,103 | 0,456 | 0,535 |
| 25 | 0,139 | 0,093 | 0,599 | 0,613 |
| 30 | 0,149 | 0,073 | 0,463 | 0,709 |
| 35 | 0,120 | 0,046 | 0,563 | 0,789 |
| 40 | 0,097 | 0,030 | 0,630 | 0,913 |

**Figuras:** [grade da época 40](../../outputs/cyclegan_crop/samples/epoch_040.png) (compare com a [do 1º treino](../../outputs/cyclegan/samples/epoch_040.png)), [curvas](../../report/figuras/cyclegan_crop_curvas.png), [amostras](../../report/figuras/cyclegan_crop_amostras.png), [piores casos](../../report/figuras/cyclegan_crop_piores.png), [vizinhos](../../report/figuras/cyclegan_crop_vizinhos.png), [memorização](../../report/figuras/cyclegan_crop_memorizacao.png). A [tabela comparativa](../../report/tabela_comparativa.md) já traz a época 40.

## O que observei (época 40)

1. **É o melhor modelo do projeto.** FID 117,4 e KID 0,036, contra 125,3 e 0,051 do primeiro treino. A volta também melhorou (`cycle_l1` 0,126 contra 0,130).
2. **Venceu o 1º treino em todos os checkpoints a partir da época 15.** Nas épocas 5 e 10 ficou atrás: com o recorte, o problema é mais difícil e o começo é mais lento.
3. **O FID oscila entre as épocas 25 e 35** (120,4 → 122,4 → 122,1) e só volta a cair na 40, com o lr perto de zero. O 1º treino teve o mesmo padrão (124,2 → 130,9 → 128,2 → 125,3). A variação de ~2 pontos entre checkpoints vizinhos é da ordem do ruído. Já a vantagem sobre o 1º treino aparece em todos os checkpoints desde a época 15.
4. **Na queda do lr, o D_pintura volta a ganhar terreno, mas bem menos.** Cai de 0,167 (época 20) para 0,097 (época 40). No 1º treino, caiu de 0,103 para 0,030. O recorte não impediu a tendência, só a atrasou: o treino termina mais ou menos onde o 1º treino estava na época 20.
5. **Sem sinal de memorização:** distância média à pintura mais próxima de 93,1, igual à distância entre pinturas reais (93,2).
6. **A diversidade caiu de 135,3 (época 20) para 130,8 (época 40)**, mas continua acima da do 1º treino (128,9) e longe de colapso (o modelo sem *cycle loss* teve 56,9).
7. **Na grade da época 40**, os céus lisos (casinha, pôr do sol) ganham pinceladas sem o padrão de grade do 1º treino. O pato branco continua visível, mas amarelado. A praia ao entardecer (2ª foto) virou uma mancha de cores fortes em que o horizonte quase some. É uma grade só, de 8 fotos.

**Sobre a previsão do autor ("vai ficar melhor"):** **confirmada** em todos os critérios escritos: FID abaixo de 125,3 ✓ (117,4), KID abaixo de 0,051 ✓ (0,036), D_pintura acima de 0,03 ✓ (0,097), diversidade perto de 137,7 ✓ (130,8, mais perto que o 1º treino).

## Teste do produto final: só o céu (2026-10-31)

Com `python -m src.galeria` (o mesmo processo do `src.stylize`: SegFormer para a máscara do céu, gerador aplicado com o lado menor em 256 e colado só no céu), em 4 fotos de `testB` com bastante céu: [`ceu.png`](../../data/treinos/treino_03_cyclegan_crop/ceu.png).

- **A máscara funcionou bem nas 4.** O templo, as palmeiras, a casinha e as árvores ficaram intactos, com borda limpa.
- **O céu pintado ainda é pouco convincente.** São manchas de cor sem pinceladas definidas. Os céus lisos (casinha, templo) viraram uma textura acinzentada e borrada, e no pôr do sol (4ª foto) apareceram listras e o sol quase sumiu. No céu da praia, a pintura ficou com vermelho e amarelo fortes, mais perto do estilo.
- **Parte do borrado é a resolução:** o gerador treinou em 128 e é aplicado em 256, e o resultado ainda é ampliado para o tamanho da foto. É o que o [treino em 256x256](2026-10-31-resolucao-256.md) testa.

## Conclusão e próximos passos

O recorte aleatório melhorou o FID em ~8 pontos e o KID em ~30%, sem perder preservação do conteúdo e sem memorização. Ele passa a ser o padrão dos próximos treinos. O tempo total foi ~10% maior, mas a 1ª metade rodou no mesmo ritmo do 1º treino, então a diferença não pode ser atribuída ao recorte (veja o diário).

- [x] Continuar até a época 40 com as mesmas opções (2026-10-31).
- [x] Ver se a queda do lr leva o FID abaixo de 125,3: sim, 117,4. O D_pintura voltou a ganhar terreno no fim, mas menos (0,097 contra 0,030).
- [ ] Próximo experimento: uma das mudanças restantes da seção 11 do notebook 03, mantendo o recorte.
