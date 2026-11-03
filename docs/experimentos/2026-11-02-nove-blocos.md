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
| 03/11 06:46 | **Interrompido a pedido do autor** (ia desligar o PC) durante a época 34. Ritmo constante de ~15,8 min por época a noite toda. Último checkpoint salvo: `epoch_030.pt` (05:45); as épocas 31–33 foram treinadas mas não salvas (só as grades `samples/epoch_031..033.png`). Nenhum checkpoint foi avaliado ainda. Para terminar: `python -m src.train_cyclegan --epochs 40 --photos 2000 --size 256 --load-size 286 --n-blocks 9 --out cyclegan_256_9blocos --resume outputs/cyclegan_256_9blocos/checkpoints/epoch_030.pt` (~2h40 para as épocas 31–40) |
| 03/11 19:10 | **Retomado** do `epoch_030.pt` pelo mesmo `rodar_treino_05.py`, que agora aceita um checkpoint como argumento (`python outputs/rodar_treino_05.py outputs/cyclegan_256_9blocos/checkpoints/epoch_030.pt`) e roda a avaliação e os gráficos no fim, como na primeira execução. Antes de retomar, o `logs.csv` foi cortado no fim da época 30 (passo 59900), porque as épocas 31–34 vão ser refeitas e o resume acrescenta linhas ao arquivo. O original foi guardado em `logs_antes_retomada.csv`. Os logs desta parte ficam em `treino_retomada.log`, `gpu_retomada.csv` e `resumo_retomada.json`. Fim do treino previsto para ~21:50 |
| 03/11 ~19:50 | **Treino encerrado de vez a pedido do autor**, que queria ver os resultados logo. Parou durante a época 32, então o modelo final é o `epoch_030.pt` (30 das 40 épocas planejadas, sem a parte final do decaimento linear da LR). O `logs.csv` foi cortado de novo na época 30. Tentei avaliar o checkpoint em paralelo com o treino, mas o autor interrompeu: a GPU não aguenta os dois juntos e o PC trava. Avaliação só com a GPU livre. Rodado só o `epoch_030` (evaluate → plots → comparar → galeria); os outros checkpoints ficam para depois, se for preciso |

## Resultado

Só o `epoch_030.pt` foi avaliado (o treino parou na época 32; ver o diário). As métricas usam as 751 fotos de `testB` contra as 400 pinturas, tudo em 256x256.

| Treino | Checkpoint | FID | Queda sobre as fotos sem conversão (195,6) | KID | Diversidade (pinturas: 289,4) | Ida e volta (L1) | Mudança foto→pintura (L1) |
|---|---|---|---|---|---|---|---|
| 04 (6 blocos) | época 30 (o melhor) | 123,5 | 36,9% | | | | |
| 04 (6 blocos) | época 40 | 124,5 | 36,3% | 0,0360 | 277,0 | 0,111 | 0,222 |
| **05 (9 blocos)** | **época 30** | **122,3** | **37,5%** | **0,0324 ± 0,0049** | **266,7** | **0,121** | **0,243** |

Distância à pintura real mais próxima: 196,8 em média (entre pinturas reais: 203,0); a menor foi 129,4. Ou seja, sem memorização.

**Figuras:** galeria [`treino_05_cyclegan_256_9blocos`](../../data/treinos/treino_05_cyclegan_256_9blocos/README.md) ([grade](../../data/treinos/treino_05_cyclegan_256_9blocos/grade.png), [céu](../../data/treinos/treino_05_cyclegan_256_9blocos/ceu.png)), [comparação com os outros treinos](../../report/figuras/comparacao_modelos.png), [curvas](../../report/figuras/cyclegan_256_9blocos_curvas.png), [piores casos](../../report/figuras/cyclegan_256_9blocos_piores.png).

## O que observei

1. **No FID, empate com o treino 04.** Na mesma época (30), foram 122,3 contra 123,5: diferença de 1,2 ponto, bem abaixo da oscilação de ~4 pontos entre checkpoints vizinhos que definimos como limite. O KID também caiu um pouco (0,0324 contra 0,0360), mas dentro do desvio (± 0,005).
2. **Nas imagens, o céu ganhou mais textura, nem sempre boa.** No céu da praia e no do templo, as pinceladas ficaram mais carregadas e com mais cores que no treino 04. No céu liso da casinha, a textura virou um padrão repetido verde-amarelado que parece mais ruído que pincelada.
3. **O resto da foto ficou parecido com o treino 04:** os detalhes sobrevivem (janelas de Bruges, templo, bola de cristal), com um tom um pouco mais amarelo/esverdeado.
4. **A transformação ficou um pouco mais forte:** a mudança foto→pintura subiu de 0,222 para 0,243 e o erro de ida e volta de 0,111 para 0,121. O gerador maior altera mais a foto e a reconstrói um pouco pior. A diversidade caiu um pouco (277,0 → 266,7), mas segue longe de colapso.

**Sobre a previsão do autor ("melhor nas imagens e no FID", FID < ~120):** **não confirmada.** O FID ficou em 122,3, empatado com o treino 04 dentro do ruído. Nas imagens, há mais textura no céu, mas não uma melhora clara. Ressalva: o modelo teve só 30 das 40 épocas, sem o final do decaimento da LR, que no treino 04 não melhorou o FID (123,5 → 124,5), mas pode deixar as texturas mais limpas.

## Conclusão e próximos passos

Com 400 pinturas, passar de 6 para 9 blocos não trouxe ganho mensurável em 30 épocas: a capacidade do gerador não parece ser o gargalo agora. O custo foi ~15,8 min por época, 27% a mais que os 12,4 min do treino 04. Para avaliar os outros checkpoints (5 a 25) e ver a curva do FID, rode `python -m src.evaluate --model cyclegan_256_9blocos --checkpoint epoch_0NN.pt` com a GPU livre.
