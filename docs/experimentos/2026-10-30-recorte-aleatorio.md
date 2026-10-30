# Experimento: recorte aleatório no aumento de dados

- **Data:** 2026-10-30 (início às 20h04)
- **Commit do código:** `3e5f8a2`
- **Comandos:**
  ```powershell
  python -m src.train_cyclegan --epochs 40 --photos 2000 --load-size 143 --out cyclegan_crop
  python -m src.evaluate --model cyclegan_crop
  python -m src.plots
  ```
- **Pasta de saída:** `outputs/cyclegan_crop/`. Logs brutos em `outputs/treino_crop/`.

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

## Resultado

_Em andamento._

## O que observei

_Em andamento._

## Conclusão e próximos passos

_Em andamento._
