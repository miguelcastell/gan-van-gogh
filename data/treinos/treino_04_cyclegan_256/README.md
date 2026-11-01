# Treino 04: `cyclegan_256`

Gerado por `python -m src.galeria`; não edite à mão (o arquivo é refeito).

Registro completo do experimento (pergunta, previsão, diário, análise): [`2026-10-31-resolucao-256.md`](../../../docs/experimentos/2026-10-31-resolucao-256.md)

## Imagens

Do checkpoint da **época 30**, o de menor FID entre os avaliados (123.5).

- [`grade.png`](grade.png): as 8 fotos fixas de teste. Linhas: original, pintada, de volta a foto.
- [`ceu.png`](ceu.png): o produto final, com só o céu pintado (máscara do SegFormer).

## Parâmetros

| Parâmetro | Valor |
|---|---|
| Resolução | 256x256 |
| Recorte aleatório | carrega em 286 e recorta 256 |
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
| FID ↓ | 124.5 (fotos sem conversão: 195.6) |
| KID ↓ | 0.0360 ± 0.0053 |
| Diversidade | 277.0 (pinturas reais: 289.4) |
| Erro de ida e volta (`cycle_l1`) ↓ | 0.111 |
| Distância à pintura mais próxima | 198.7 (entre pinturas reais: 203.0) |

Em 256x256: o FID só é comparável entre treinos da mesma resolução.
