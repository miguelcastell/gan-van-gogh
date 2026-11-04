# Galeria dos treinos

Uma pasta por treino, em ordem cronológica. Gerado por `python -m src.galeria`; a comparação lado a lado de todos está em [`report/figuras/comparacao_modelos.png`](../../report/figuras/comparacao_modelos.png).

| Treino | Resolução | Melhor época | FID | Céu pintado |
|---|---|---|---|---|
| [01 `cyclegan`](treino_01_cyclegan/README.md) | 128px | 25 | 124.2 | [ceu.png](treino_01_cyclegan/ceu.png) |
| [02 `cyclegan_sem_cycle`](treino_02_cyclegan_sem_cycle/README.md) | 128px | 15 | 140.5 | [ceu.png](treino_02_cyclegan_sem_cycle/ceu.png) |
| [03 `cyclegan_crop`](treino_03_cyclegan_crop/README.md) | 128px | 40 | 117.4 | [ceu.png](treino_03_cyclegan_crop/ceu.png) |
| [04 `cyclegan_256`](treino_04_cyclegan_256/README.md) | 256px | 30 | 123.5 | [ceu.png](treino_04_cyclegan_256/ceu.png) |
| [05 `cyclegan_256_9blocos`](treino_05_cyclegan_256_9blocos/README.md) (em andamento) | 256px | 30 | 122.3 | [ceu.png](treino_05_cyclegan_256_9blocos/ceu.png) |
| [06 `cyclegan_256_ceu`](treino_06_cyclegan_256_ceu/README.md) | 256px | 40 | 127.1 | [ceu.png](treino_06_cyclegan_256_ceu/ceu.png) |

O FID só é comparável entre treinos da mesma resolução.
