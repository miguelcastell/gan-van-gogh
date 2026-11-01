# Galeria dos treinos

Uma pasta por treino, em ordem cronológica. Gerado por `python -m src.galeria`; a comparação lado a lado de todos está em [`report/figuras/comparacao_modelos.png`](../../report/figuras/comparacao_modelos.png).

| Treino | Resolução | Melhor época | FID | Céu pintado |
|---|---|---|---|---|
| [01 `cyclegan`](treino_01_cyclegan/README.md) | 128px | 25 | 124.2 | [ceu.png](treino_01_cyclegan/ceu.png) |
| [02 `cyclegan_sem_cycle`](treino_02_cyclegan_sem_cycle/README.md) | 128px | 15 | 140.5 | [ceu.png](treino_02_cyclegan_sem_cycle/ceu.png) |
| [03 `cyclegan_crop`](treino_03_cyclegan_crop/README.md) | 128px | 40 | 117.4 | [ceu.png](treino_03_cyclegan_crop/ceu.png) |

O FID só é comparável entre treinos da mesma resolução.
