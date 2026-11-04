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

## Resultado

(preencher no fim)

## O que observei

(preencher no fim)

## Conclusão e próximos passos

(preencher no fim)
