# Experimento: 256x256 com 9 blocos residuais (treino 05)

- **Data:** 2026-11-02
- **Commit do código:** `4edd4bf` (nenhuma mudança de código: `--n-blocks` já existia)
- **Comandos** (rodados em sequência por um script, durante a noite):
  ```powershell
  python -m src.train_cyclegan --epochs 40 --photos 2000 --size 256 --load-size 286 --n-blocks 9 --out cyclegan_256_9blocos
  python -m src.evaluate --model cyclegan_256_9blocos --checkpoint epoch_0NN.pt   # para cada checkpoint (5, 10, ..., 40)
  python -m src.evaluate --model cyclegan_256_9blocos                             # deixa o metrics.json na época 40
  python -m src.plots
  python -m src.comparar
  python -m src.galeria
  ```
- **Pasta de saída:** `outputs/cyclegan_256_9blocos/`. Logs brutos em `outputs/treino_256_9blocos/`.

## Pergunta

O [treino 04, em 256x256](2026-10-31-resolucao-256.md), deu as primeiras pinceladas reconhecíveis, mas o gerador continua com 6 blocos residuais, a versão que o artigo da CycleGAN usa para 128x128. Para 256x256, o artigo usa **9 blocos**.

O objetivo do projeto é que o gerador transforme **qualquer foto** numa pintura de Van Gogh com pinceladas visíveis. Os blocos residuais são a parte do gerador que faz a transformação de verdade. Eles trabalham em 64x64 com 256 canais (com entrada de 256), entre o *encoder* que reduz a imagem e o *decoder* que a reconstrói. Cada bloco a mais:
- dá mais capacidade, ou seja, mais parâmetros para representar o estilo;
- aumenta o **campo receptivo**: cada pixel da saída "enxerga" uma região maior da foto. Cada bloco tem duas convs 3x3, então soma ~4 pixels de 64x64, o que dá ~16 pixels na imagem original. Com 3 blocos a mais, cada pixel enxerga ~48 pixels a mais da foto, o que ajuda a manter pinceladas coerentes em regiões grandes, como o céu.

Mudança única em relação ao treino 04: `--n-blocks 9` em vez de 6. Todo o resto fica igual.

## Previsão (antes de rodar)

Do autor, escolhida antes do treino: **"Melhor nas imagens e no FID"**, ou seja, pinceladas mais marcadas e FID abaixo de ~120 (queda relativa maior que os 36,3% do treino 04).

Para conferir no fim:
- FID no melhor checkpoint abaixo de ~120 e queda relativa (sobre as fotos sem conversão em 256, 195,6) maior que 36,3%. Como a oscilação entre checkpoints vizinhos no treino 04 foi de ~4 pontos (123,5 → 127,9 → 124,5), uma diferença menor que isso não conta como melhora;
- pinceladas mais marcadas na grade e no céu, comparando com o treino 04 na [comparação lado a lado](../../report/figuras/comparacao_modelos.png);
- diversidade sem colapso e sem memorização (distância à pintura mais próxima perto da distância entre pinturas reais);
- o risco contrário: com só 400 pinturas, um gerador maior pode não ajudar, e o D_pintura pode voltar a dominar.

## Configuração

Igual ao treino 04; só o número de blocos muda.

| Parâmetro | Valor |
|---|---|
| Épocas / fotos por época | 40 / 2.000 (80.000 passos) |
| Resolução | 256x256, recortada de 286x286 |
| λ_cycle / λ_identity | 10 / 5 |
| Queda do lr | a partir da época 20 |
| Precisão mista (`--amp`) | não (ligar mudaria duas coisas ao mesmo tempo) |
| Seed | 42 |
| **Blocos residuais** | **9** (antes: 6) |

**Teste de custo** (300 passos, 2026-11-02 ~21:55, pasta apagada depois): **~493 ms por passo**, contra ~391 ms com 6 blocos (+26%). Estimativa: ~16,4 min por época, **~10h50** no total. Pico de VRAM: 3,8 GB dos 6 GB (amostra a cada 3 s, só 300 passos).

**Tropeço no teste:** pelo Git Bash, `python` era o Python do sistema (sem CUDA) e o `print` do `λ` quebrava com `UnicodeEncodeError` (terminal em cp1252). A solução foi chamar `.venv\Scripts\python.exe` direto, com `PYTHONUTF8=1`.

## Dados

Os mesmos do treino 04: as 400 pinturas de `trainA` e 2.000 fotos de `trainB` sorteadas com a seed 42 (cache `data/vangogh2photo_286/`). As grades por época usam as 8 primeiras fotos de `testB`, e a avaliação usa as 751 fotos de `testB` contra as 400 pinturas, tudo em 256x256.

## Diário da execução

| Hora | Evento |
|---|---|
| 21:52 | Início, pelo script `outputs/rodar_treino_05.py` (treino → avaliação de cada checkpoint → `plots` → `comparar` → `galeria`), num processo separado. GPU registrada a cada 30 s em `gpu.csv` |
| 22:08 | Época 1 em ~16 min (grade `samples/epoch_001.png` salva), no ritmo estimado: fim do treino previsto para ~08:45 de 03/11. GPU a 100%, 3,8 GB de VRAM, 74 °C. O `treino.log` e o `gpu.csv` ficam vazios durante o treino (a saída redirecionada para arquivo só é gravada em blocos); as losses por passo estão em `outputs/cyclegan_256_9blocos/logs.csv` |

## Resultado

(preencher no fim)

## O que observei

(preencher no fim)

## Conclusão e próximos passos

(preencher no fim)
