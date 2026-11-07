# Experimento: continuar o treino 06 até a época 80, com λ_identity 5 e 1 (treinos 07 e 08)

- **Data:** 2026-11-06 (início às 23:27)
- **Commit do código:** `935b0d9` + a correção do LR no `--resume` em `src/train_cyclegan.py` (sem commit no início do treino)
- **Comandos:**
  ```powershell
  python outputs/rodar_treinos_07_08.py   # treino 07 → avaliação → treino 08 → avaliação → plots → comparar → galeria
  # os treinos em si (os dois partem do checkpoint da época 40 do treino 06):
  python -m src.train_cyclegan --epochs 80 --decay-start 0 --size 256 --load-size 286 --n-blocks 6 --photos-list data/selecao_ceu/selecionadas.txt --resume outputs/cyclegan_256_ceu/checkpoints/epoch_040.pt --lambda-identity 5 --out cyclegan_256_ceu_80
  python -m src.train_cyclegan ... (igual) ... --lambda-identity 1 --out cyclegan_256_ceu_id1
  ```
- **Pastas de saída:** `outputs/cyclegan_256_ceu_80/` (treino 07) e `outputs/cyclegan_256_ceu_id1/` (treino 08). Logs brutos em `outputs/treino_07_08/`.

## Pergunta

1. **Treino 07:** no treino 06, o FID ainda caía na época 40 (139,6 → 129,8 → 127,1 nas épocas 30, 35 e 40). Faltava treino? Mais 40 épocas melhoram o céu?
2. **Treino 08:** com o peso da *identity loss* menor (1 em vez de 5), o gerador transforma mais a foto? Isso resolve o céu liso, que hoje vira só textura sem pinceladas (o ponto fraco que aparece na Estátua da Liberdade)?

Os dois partem do mesmo ponto (época 40 do treino 06) e com a mesma programação de LR, então a única diferença entre o 07 e o 08 é o λ_identity. Assim, o efeito dele fica isolado. Alternativa descartada: treinar o 08 do zero até a época 80, que levaria ~12h40 em vez de ~6h20 e misturaria o efeito do λ_identity com o de começar do zero.

## Previsão (antes de rodar)

**Previsão do autor:** sem previsão registrada. O autor disse só o que espera do resultado: "qro q fique bonito" (2026-11-06).

Hipótese técnica do Claude:
- **07:** o FID cai mais alguns pontos (talvez para perto de 120) e o céu fica um pouco mais coerente. Mas as mudanças devem ser pequenas, porque os treinos 04 a 06 já mostraram um teto com 400 pinturas. Nas épocas 41 a 45, pode haver uma piora passageira: o LR volta de 0 para 1e-4.
- **08:** a *identity loss* pune o gerador quando ele muda uma pintura que já é pintura (e, do outro lado, uma foto que já é foto). Na prática, ela segura as cores perto do original. Com peso menor, espero cores mais distantes da foto e céu liso mais transformado, com mudança foto→pintura (L1) maior que 0,239. O risco é a cor errar: céu verde, sol mais apagado ou tom geral deslocado. O FID pode até melhorar, porque o FID gosta de imagens que "parecem" mais pintura.

## Configuração

Igual ao [treino 06](2026-11-03-fotos-de-ceu.md), exceto:

| Parâmetro | Treino 07 | Treino 08 |
|---|---|---|
| Ponto de partida | época 40 do treino 06 (pesos, otimizadores) | igual |
| Épocas | 41 a 80 (40 a mais) | igual |
| LR | 1e-4 na época 41, caindo em linha reta até 0 na 80 (`--decay-start 0`, fator 1 − época/80) | igual |
| λ_cycle / λ_identity | 10 / **5** | 10 / **1** |
| Seed | 42 | 42 |

**Por que esse LR:** com `--epochs 80` e o `--decay-start` padrão (metade = 40), o LR voltaria para o máximo (2e-4) na época 41. Um salto tão grande poderia "desarrumar" o que o modelo já aprendeu. Com `--decay-start 0`, ele recomeça na metade (1e-4) e cai até 0, como se o treino de 80 épocas tivesse decaído desde o início.

Observação: o `args` salvo nos checkpoints desses treinos diz `epochs 80, decay_start 0`, mas as épocas 1 a 40 usaram a programação do treino 06 (`epochs 40, decay_start 20`). O README da galeria mostra o `args` do checkpoint. O `logs.csv` das épocas 1 a 40 foi copiado do treino 06, para as curvas ficarem contínuas.

## Problema resolvido antes de rodar: o LR na retomada

O `--resume` tinha um bug que teria estragado a época 41. O otimizador volta do checkpoint com o LR da época em que foi salvo, que no fim do treino 06 era **0**. O `LambdaLR` só recalcula o LR no fim de cada época (`sched.step()`). Então, a época 41 inteira rodaria com LR 0 e o Adam não mudaria nenhum peso. Só a partir da 42 o LR iria para o valor certo.

A correção em `src/train_cyclegan.py`: logo depois de carregar o checkpoint, o LR de cada otimizador é recalculado pela programação atual (`base_lr × lr_factor(last_epoch)`). O script agora imprime o LR na retomada. Numa retomada comum (mesmo `--epochs`), o valor recalculado é igual ao salvo, então nada muda nesse caso.

## Dados

Igual ao treino 06: 400 pinturas de `trainA` e as 1.500 fotos de `trainB` em `data/selecao_ceu/selecionadas.txt`. Avaliação nas 751 fotos de `testB`.

## Diário da execução

| Hora | Evento |
|---|---|
| 06/11 23:27 | Início pelo Claude Code, em segundo plano, a pedido do autor. RAM livre: 7,0 de 15,7 GB; GPU livre. Previsão: ~6h20 por treino + ~10 min de avaliação de cada, fim por volta das 12:30 de 07/11 |
| 06/11 23:35 | Medido: ~384 ms por passo (500 passos em 192 s), o mesmo do treino 06, ou seja, 9,6 min por época. LR conferido: o otimizador está em 1e-4 na época 41. A coluna `lr` do `logs.csv` mostra 0 na época 41 porque registra `sched.get_last_lr()`, que vem do checkpoint e só é atualizado no fim da época; da 42 em diante ela mostra o valor certo. Os `print` do treino só aparecem no `.log` em blocos, porque o Python guarda a saída num buffer quando ela vai para um arquivo |
| 07/11 05:46 | Treino 07 concluído sem interrupção: **378,6 min** (6h19). Avaliação dos 8 checkpoints em ~1 min cada, todas com código de saída 0 |
| 07/11 ~05:55 | **Script encerrado pelo Claude Code por falta de RAM no sistema**, logo no início do treino 08, provavelmente enquanto carregava as fotos. O treino 08 não salvou nenhum checkpoint. O 07 já estava completo e avaliado. Os passos `plots`, `comparar` e `galeria` foram rodados depois, à mão, todos com código de saída 0 |
| 07/11 | Treino 08 **pendente**: só será reiniciado quando o autor pedir. Recomendação: rodar num PowerShell próprio, fora do Claude Code, com o navegador fechado |

## Resultado (treino 07)

Métricas nas 751 fotos de `testB` contra as 400 pinturas, em 256x256. Valores de `outputs/cyclegan_256_ceu_80/por_checkpoint.csv` e `metrics.json`.

**FID por checkpoint** (épocas 5 a 40: as do [treino 06](2026-11-03-fotos-de-ceu.md)):

| Época | 40 | 45 | 50 | 55 | 60 | 65 | 70 | 75 | 80 |
|---|---|---|---|---|---|---|---|---|---|
| FID | 127,1 | **119,2** | 125,8 | 123,3 | 126,0 | 125,8 | 128,0 | 126,5 | 125,0 |
| KID | 0,0356 | **0,0279** | 0,0345 | 0,0321 | 0,0371 | 0,0361 | 0,0397 | 0,0368 | 0,0358 |
| Ida e volta (L1) | 0,127 | 0,135 | 0,132 | 0,128 | 0,133 | 0,127 | 0,125 | 0,124 | 0,124 |

**Época 80:** FID 125,0 (queda de 36,1% sobre as fotos sem conversão, 195,6), KID 0,0358 ± 0,0052, diversidade 276,9 (pinturas: 289,4), ida e volta 0,124, mudança foto→pintura 0,240. Distância à pintura real mais próxima: 194,9 em média (entre pinturas reais: 203,0), mínima de 131,3, então não há memorização.

**Figuras:** galeria [`treino_07_cyclegan_256_ceu_80`](../../data/treinos/treino_07_cyclegan_256_ceu_80/README.md) (usa a época 45, a de menor FID; [céu](../../data/treinos/treino_07_cyclegan_256_ceu_80/ceu.png)), [comparação com os outros treinos](../../report/figuras/comparacao_modelos.png) (usa a época 80), [curvas](../../report/figuras/cyclegan_256_ceu_80_curvas.png).

## O que observei (treino 07)

Observações do Claude. A avaliação visual do autor ainda está pendente.

1. **A época 45 tem o menor FID de todos os treinos em 256 (119,2)**, mas é um ponto isolado. Da 50 à 80, o FID oscila entre 123 e 128, sem tendência de queda. A diferença entre a 45 e a 80 (6 pontos) é maior que a oscilação usual (~4), mas não muito. A hipótese "faltava treino" se confirma pouco: mais 40 épocas não trouxeram queda contínua.
2. **Céu da galeria, época 45 contra o treino 06 (época 40):**
   - *Pôr do sol na praia:* tons mais quentes (laranja e amarelo) e menos fragmentado que no 06. Parece menos caótico.
   - *Céu liso da casinha:* azul mais definido e um pouco mais limpo. Continua uma textura, sem pinceladas.
   - *Nuvens do templo:* traços alongados, agora mais verdes (menos cáqui). A cor ainda foge da paleta de Van Gogh.
   - *Sol sobre o morro:* o disco reaparece como uma mancha clara rosada, mas a faixa horizontal no meio continua.
3. **Na comparação (época 80), o 07 é muito parecido com o 06.** As diferenças são pequenas: céu da casinha um pouco mais azul, água dos patos ainda turquesa e pouco transformada.
4. **A ida e volta (L1) caiu devagar** (0,135 → 0,124): o modelo reconstrói melhor a foto original, mas isso não aparece como mudança visível.

**Sobre a previsão:** não havia previsão do autor. A hipótese técnica ("FID perto de 120, mudanças pequenas") se confirmou na época 45 e em parte na 80. A "piora passageira nas épocas 41 a 45" não aconteceu: a 45 foi a melhor.

## Conclusão e próximos passos

Treino 07: mais épocas deram mudanças pequenas. O melhor ponto foi logo no começo (época 45, FID 119,2), e depois o FID estabilizou em ~125, como nos treinos 04 a 06. O teto com 400 pinturas continua.

- **A decidir pelo autor:** comparar visualmente o treino 06 (época 40, o modelo atual da entrega) com o 07 (época 45 e 80). Se gostar mais do 07, registrar num novo ADR que substitui o [ADR 0010](../adr/0010-treino-06-como-modelo-da-entrega.md).
- **Treino 08 (λ_identity 1):** ainda não rodou. Os resultados dele entram neste mesmo registro.
