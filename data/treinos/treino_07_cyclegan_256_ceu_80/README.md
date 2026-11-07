# Treino 07: `cyclegan_256_ceu_80`

Gerado por `python -m src.galeria`; não edite à mão (o arquivo é refeito).

Registro completo do experimento (pergunta, previsão, diário, análise): [`2026-11-06-oitenta-epocas-e-identity.md`](../../../docs/experimentos/2026-11-06-oitenta-epocas-e-identity.md)

## Imagens

Do checkpoint da **época 45**, o de menor FID entre os avaliados (119.2).

- [`grade.png`](grade.png): as 8 fotos fixas de teste. Linhas: original, pintada, de volta a foto.
- [`ceu.png`](ceu.png): o produto final, com só o céu pintado (máscara do SegFormer).

## Parâmetros

| Parâmetro | Valor |
|---|---|
| Resolução | 256x256 |
| Recorte aleatório | carrega em 286 e recorta 256 |
| Épocas / fotos por época | 80 / None |
| Blocos residuais no gerador | 6 |
| *Cycle loss* | ligada |
| λ_cycle / λ_identity | 10.0 / 5.0 |
| lr / beta1 | 0.0002 / 0.5 |
| Queda do lr | a partir da época 40 |
| Batch / image pool | 1 / 50 |
| Precisão mista (`--amp`) | não |
| Seed | 42 |

## Métricas (`epoch_080.pt`)

| Métrica | Valor |
|---|---|
| FID ↓ | 125.0 (fotos sem conversão: 195.6) |
| KID ↓ | 0.0358 ± 0.0052 |
| Diversidade | 276.9 (pinturas reais: 289.4) |
| Erro de ida e volta (`cycle_l1`) ↓ | 0.124 |
| Distância à pintura mais próxima | 194.9 (entre pinturas reais: 203.0) |

Em 256x256: o FID só é comparável entre treinos da mesma resolução.
