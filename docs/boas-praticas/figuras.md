# Figuras e imagens de cada treino

Regras para que todo treino deixe imagens que permitam ver, a olho, se ele melhorou ou não em relação aos anteriores. Combinadas em 2026-10-31.

## O que todo treino deve deixar salvo

| Imagem | Onde | Quem gera |
|---|---|---|
| Grade das mesmas 8 fotos de teste ao fim de cada época | `outputs/<treino>/samples/epoch_XXX.png` (no git, só as de 5 em 5) | `src.train_cyclegan`, automático |
| GIF da evolução | `outputs/<treino>/evolucao.gif` (fora do git) | `src.train_cyclegan`, automático |
| Amostras, piores casos, vizinhos e memorização | `report/figuras/<treino>_*.png` | `src.evaluate` |
| **Todos os treinos lado a lado, mesmas fotos** | `report/figuras/comparacao_modelos.png` | `python -m src.comparar` |

## Regras

- **Ao fim de todo treino, rode `python -m src.comparar` depois do `src.evaluate`.** A figura é refeita com todos os treinos, então é nela que se vê se o novo é melhor que os anteriores.
- **Se outro treino estiver rodando, use `--cpu`.** Na GPU de 6 GB, um treino em 256x256 já usa ~5,3 GB, e disputar memória pode derrubá-lo.
- **As fotos de comparação são sempre as 8 primeiras de `testB`**, as mesmas das grades por época. Assim qualquer grade de qualquer treino é comparável com qualquer outra.
- **O registro do experimento linka a grade da última época e a `comparacao_modelos.png`**, e descreve o que se vê nelas. A figura é refeita a cada treino, então o texto é o que preserva a impressão daquele momento.
- **A imagem não substitui as métricas.** São só 8 fotos: elas mostram defeitos (grade, colapso, cores erradas), mas a conclusão vem do FID e do KID nas 751 fotos.
