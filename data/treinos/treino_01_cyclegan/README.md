# Treino 01: `cyclegan`

Gerado por `python -m src.galeria`; não edite à mão (o arquivo é refeito).

Registro completo do experimento (pergunta, previsão, diário, análise): [`2026-10-29-primeiro-treino.md`](../../../docs/experimentos/2026-10-29-primeiro-treino.md)

## Imagens

Do checkpoint da **época 25**, o de menor FID entre os avaliados (124.2).

- [`grade.png`](grade.png): as 8 fotos fixas de teste. Linhas: original, pintada, de volta a foto.
- [`ceu.png`](ceu.png): o produto final, com só o céu pintado (máscara do SegFormer).

## Parâmetros

| Parâmetro | Valor |
|---|---|
| Resolução | 128x128 |
| Recorte aleatório | não (só espelhamento) |
| Épocas / fotos por época | 40 / 2000 |
| Blocos residuais no gerador | 6 |
| *Cycle loss* | ligada |
| λ_cycle / λ_identity | 10.0 / 5.0 |
| lr / beta1 | 0.0002 / 0.5 |
| Queda do lr | a partir da época 20 |
| Batch / image pool | 1 / 50 |
| Precisão mista (`--amp`) | não |
| Seed | 42 |

## Métricas (`epoch_040.pt`)

| Métrica | Valor |
|---|---|
| FID ↓ | 125.3 (fotos sem conversão: 179.8) |
| KID ↓ | 0.0514 ± 0.0054 |
| Diversidade | 128.9 (pinturas reais: 137.7) |
| Erro de ida e volta (`cycle_l1`) ↓ | 0.130 |
| Distância à pintura mais próxima | 93.1 (entre pinturas reais: 93.2) |

Em 128x128: o FID só é comparável entre treinos da mesma resolução.
