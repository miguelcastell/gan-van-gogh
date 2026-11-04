# Experimento: treinar só com fotos de céu marcado (treino 06)

- **Data:** 2026-11-03
- **Commit do código:** `d2532cc` + `src/selecionar_fotos.py` e a opção `--photos-list` em `src/train_cyclegan.py` (ainda sem commit no início do treino)
- **Comandos:**
  ```powershell
  python -m src.selecionar_fotos --n 1500
  python outputs/rodar_treino_06.py   # treino → avaliação de cada checkpoint → plots → comparar → galeria
  # o treino em si:
  python -m src.train_cyclegan --epochs 40 --size 256 --load-size 286 --n-blocks 6 --out cyclegan_256_ceu --photos-list data/selecao_ceu/selecionadas.txt
  ```
- **Pasta de saída:** `outputs/cyclegan_256_ceu/`. Logs brutos em `outputs/treino_256_ceu/`.

## Pergunta

Se o gerador treinar só com fotos de céu marcado (pôr do sol, nuvens), o céu pintado fica mais bonito e mais parecido com Van Gogh que no [treino 04](2026-10-31-resolucao-256.md)? Motivo da mudança: [ADR 0008](../adr/0008-treinar-com-fotos-de-ceu.md).

## Previsão (antes de rodar)

Sem previsão do autor registrada. Hipótese técnica: o céu na [galeria](../../data/treinos/README.md) e no `stylize` fica mais carregado e coerente que no treino 04, porque toda a capacidade do gerador vai para céus. O FID em todo o `testB` pode ficar igual ou pior, porque metade do `testB` tem pouco céu, e o gerador viu menos florestas e cidades.

## Configuração

Igual ao treino 04, exceto as fotos.

| Parâmetro | Valor |
|---|---|
| Épocas / fotos por época | 40 / 1.500 (as selecionadas, todas por época) |
| Tamanho | 256, com recorte aleatório de 286 |
| Blocos residuais | 6 |
| λ_cycle / λ_identity | 10 / 5 |
| Seed | 42 |

## Dados

400 pinturas de `trainA` e as 1.500 fotos de `trainB` listadas em `data/selecao_ceu/selecionadas.txt`. Critério ([ADR 0008](../adr/0008-treinar-com-fotos-de-ceu.md)): fração de céu ≥ 25% (4.090 das 6.287 fotos passaram) e maior nota de colorido + variação de luz no céu. Conferência visual: [`amostra.png`](../../data/selecao_ceu/amostra.png). Avaliação nas 751 fotos de `testB`, como nos treinos anteriores.

## Diário da execução

| Hora | Evento |
|---|---|
| 03/11 ~20:45 | Início. GPU a 98%, 2,9 GB de VRAM no primeiro passo. Previsão: 60.000 passos a ~400 ms, ~6h40, fim por volta das 03:30 de 04/11 |
| 03/11 ~20:58 | **Interrompido** na época 2, passo 1800, sem nenhum checkpoint salvo (o primeiro seria o da época 5). Não foi erro do treino: o Claude Code encerra processos em segundo plano quando a RAM do sistema fica criticamente baixa (5,1 de 15,7 GB livres logo depois). Para não acontecer de novo, o treino deve rodar num terminal próprio, fora do Claude Code |
| 03/11 22:46 | **Reiniciado do zero** pelo `rodar_treino_06.py`, de novo pelo Claude Code em segundo plano, a pedido do autor ("não vai mais se encerrar"). Nenhum checkpoint da tentativa anterior foi aproveitado. RAM livre no início: 5,7 de 15,7 GB |
| 04/11 05:05 | Treino concluído sem interrupção: **379,6 min** (6h20), 9,5 min por época, ~380 ms por passo (o mesmo passo do treino 04; a época é mais curta porque tem 1.500 fotos em vez de 2.000) |
| 04/11 05:14 | Avaliação dos 8 checkpoints (~1 min cada), plots, `comparar` e `galeria` concluídos, todos com código de saída 0 (`outputs/treino_256_ceu/resumo.json`) |

## Resultado

As métricas usam as 751 fotos de `testB` contra as 400 pinturas, tudo em 256x256, como nos treinos 04 e 05. Valores de `outputs/treino_256_ceu/por_checkpoint.csv` e `outputs/treino_256/por_checkpoint.csv`.

**FID por checkpoint:**

| Época | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|
| Treino 04 (2.000 fotos sorteadas) | 160,4 | 141,6 | 142,0 | 131,5 | 129,1 | **123,5** | 127,9 | 124,5 |
| **Treino 06 (1.500 fotos de céu)** | 153,9 | 143,2 | 138,0 | 144,1 | 130,4 | 139,6 | 129,8 | **127,1** |

**Melhor checkpoint de cada um:**

| Treino | Checkpoint | FID | Queda sobre as fotos sem conversão (195,6) | KID | Diversidade (pinturas: 289,4) | Ida e volta (L1) | Mudança foto→pintura (L1) |
|---|---|---|---|---|---|---|---|
| 04 | época 30 (o melhor) | 123,5 | 36,9% | 0,0330 | 284,6 | 0,117 | |
| 04 | época 40 | 124,5 | 36,3% | 0,0360 | 277,0 | 0,111 | 0,222 |
| **06** | **época 40 (o melhor)** | **127,1** | **35,0%** | **0,0356 ± 0,0053** | **281,8** | **0,127** | **0,239** |

Distância à pintura real mais próxima: 197,0 em média (entre pinturas reais: 203,0). A menor foi 132,8, então não há memorização.

**Figuras:** galeria [`treino_06_cyclegan_256_ceu`](../../data/treinos/treino_06_cyclegan_256_ceu/README.md) ([grade](../../data/treinos/treino_06_cyclegan_256_ceu/grade.png), [céu](../../data/treinos/treino_06_cyclegan_256_ceu/ceu.png)), [comparação com os outros treinos](../../report/figuras/comparacao_modelos.png), [curvas](../../report/figuras/cyclegan_256_ceu_curvas.png), [piores casos](../../report/figuras/cyclegan_256_ceu_piores.png).

## O que observei

Observações do Claude ao comparar o `ceu.png` do treino 06 com o do treino 04 e a `comparacao_modelos.png`. O autor ainda não viu as imagens.

1. **No FID, empate ou leve piora.** O melhor checkpoint ficou em 127,1, contra 123,5 do treino 04. A diferença de 3,6 pontos está no limite da oscilação de ~4 pontos entre checkpoints vizinhos. O KID da época 40 é praticamente igual (0,0356 contra 0,0360). Era o que a hipótese técnica esperava, porque metade do `testB` tem pouco céu.
2. **A curva do FID oscilou mais e ainda caía no fim** (139,6 → 129,8 → 127,1 nas épocas 30, 35 e 40). No treino 04, a melhora tinha parado na época 30. Com menos fotos por época, o treino 06 deu 60.000 passos contra 80.000 do treino 04, e talvez ainda não tivesse estabilizado.
3. **O céu não ficou claramente melhor; ficou diferente:**
   - *Pôr do sol na praia:* mais saturado e mais fragmentado, com manchas vermelhas, amarelas e azuis contornadas de escuro. Parece mais "pintado" que no treino 04, mas também mais caótico.
   - *Céu liso da casinha:* uma textura marmorizada azul-clara e esbranquiçada, mais pálida que o azul com traços escuros do treino 04. Continua sem pinceladas, a mesma limitação dos treinos anteriores em céu liso.
   - *Nuvens do templo:* traços alongados que seguem as nuvens, mas num tom verde-cáqui. O treino 04 usou azul e verde-água, mais perto da paleta de Van Gogh.
   - *Sol sobre o morro:* o disco do sol quase sumiu e apareceram faixas horizontais. O treino 04 manteve o sol visível.
4. **Fora do céu, transformou menos em algumas cenas.** Na foto dos patos (água, quase sem céu), a água continuou turquesa e quase fotográfica, enquanto os treinos 03 a 05 a deixaram amarelada e pintada. É a consequência prevista no ADR 0008: o gerador viu menos cenas sem céu.
5. **Os números de conteúdo pioraram um pouco:** ida e volta de 0,111 para 0,127 e mudança foto→pintura de 0,222 para 0,239. A diversidade ficou estável (277,0 → 281,8).

**Sobre a previsão:** não havia previsão do autor. A hipótese técnica era "céu mais carregado e coerente; FID igual ou pior". **Confirmada em parte:** o FID ficou igual ou um pouco pior, e o céu ficou mais carregado no pôr do sol, mas não mais coerente. Nas nuvens e no sol, o treino 04 está melhor.

## Conclusão e próximos passos

Treinar só com fotos de céu marcado não trouxe ganho claro nem nas métricas nem nas imagens do céu. A mudança deixou o estilo mais intenso no pôr do sol, mas errou mais nas cores das nuvens e apagou o sol. É o terceiro treino seguido (04, 05 e 06) com resultado empatado. Com 400 pinturas e ~40 épocas, o modelo parece ter chegado ao limite do que consegue fazer; o próximo ganho deve vir da escolha das fotos de entrada e do `--size` no `stylize` (itens 2 e 3 em [03-proximos-passos](../guia/03-proximos-passos.md#como-melhorar-as-imagens-levantado-em-2026-11-03)).

Sugestões, a decidir pelo autor:
- **Modelo principal da entrega:** manter o treino 04 (época 30). O treino 06 pode aparecer nos slides como experimento de dados, com o pôr do sol da praia como o caso em que ele vai melhor.
- **Teste barato antes de descartar o treino 06:** rodar `python -m src.stylize <foto> --model cyclegan_256_ceu` e o mesmo com `cyclegan_256` nas fotos da demonstração. A galeria tem só 4 céus.
- **Sobre o ponto 2:** como o FID ainda caía na época 40, mais épocas poderiam ajudar, mas não vale a GPU agora, porque nada indica que o céu fique melhor que o do treino 04.
