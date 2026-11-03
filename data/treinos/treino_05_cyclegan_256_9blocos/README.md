# Treino 05: `cyclegan_256_9blocos`

Gerado por `python -m src.galeria`; não edite à mão (o arquivo é refeito).

**Treino em andamento:** as imagens são do último checkpoint salvo até agora.

Registro completo do experimento (pergunta, previsão, diário, análise): [`2026-11-02-nove-blocos.md`](../../../docs/experimentos/2026-11-02-nove-blocos.md)

## Imagens

Do checkpoint da **época 30**, o de menor FID entre os avaliados (122.3).

- [`grade.png`](grade.png): as 8 fotos fixas de teste. Linhas: original, pintada, de volta a foto.
- [`ceu.png`](ceu.png): o produto final, com só o céu pintado (máscara do SegFormer).

## Parâmetros

| Parâmetro | Valor |
|---|---|
| Resolução | 256x256 |
| Recorte aleatório | carrega em 286 e recorta 256 |
| Épocas / fotos por época | 40 / 2000 |
| Blocos residuais no gerador | 9 |
| *Cycle loss* | ligada |
| λ_cycle / λ_identity | 10.0 / 5.0 |
| lr / beta1 | 0.0002 / 0.5 |
| Queda do lr | a partir da época 20 |
| Batch / image pool | 1 / 50 |
| Precisão mista (`--amp`) | não |
| Seed | 42 |

## Métricas (`epoch_030.pt`)

| Métrica | Valor |
|---|---|
| FID ↓ | 122.3 (fotos sem conversão: 195.6) |
| KID ↓ | 0.0324 ± 0.0049 |
| Diversidade | 266.7 (pinturas reais: 289.4) |
| Erro de ida e volta (`cycle_l1`) ↓ | 0.121 |
| Distância à pintura mais próxima | 196.8 (entre pinturas reais: 203.0) |

Em 256x256: o FID só é comparável entre treinos da mesma resolução.
