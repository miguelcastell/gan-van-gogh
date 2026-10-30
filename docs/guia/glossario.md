# Glossário

Termos explicados ao longo do projeto, em ordem alfabética. Ao aprender um termo novo, adicione aqui.

## Ablação (estudo de ablação)

Tirar uma peça do modelo e treinar de novo, mudando **só ela**, para medir o quanto essa peça contribui. A comparação "CycleGAN completa x sem *cycle loss*" é uma ablação.

Onde aparece: [primeiro treino](../experimentos/2026-10-29-primeiro-treino.md), seção 1 do notebook 03.

## Artefato de tabuleiro (*checkerboard artifact*)

Padrão de grade ou listras que aparece em imagens geradas por convoluções transpostas, as camadas que aumentam a imagem no decodificador. Ele aparece mais em áreas lisas, como céus (Odena et al., 2016). A correção comum é trocar a convolução transposta por *upsample* + convolução.

Onde aparece: o céu da casinha nas grades do primeiro treino.

## Colapso de modo (*mode collapse*)

Quando o gerador encontra uma saída que engana o discriminador e passa a produzi-la para quase qualquer entrada, perdendo a variedade. No primeiro treino sem *cycle loss*, 7 das 8 fotos fixas viraram a mesma paisagem na época 40.

Como detectar: a **diversidade** das saídas despenca, e o FID piora. As losses adversariais podem parecer normais.

## Diversidade

A distância média entre todos os pares de imagens de um conjunto. É a medida usada no projeto para detectar colapso de modo, já que não há classes. Ela é comparada com a diversidade das pinturas reais (137,7). Veja a [ficha de métricas](../referencia/metricas.md).

## Esteganografia (na CycleGAN)

A capacidade de a CycleGAN esconder informação da foto em sinais quase invisíveis dentro da pintura, só para conseguir reconstruí-la na volta (Chu et al., 2017). Para testar, some um ruído pequeno à pintura: se a reconstrução desmoronar, a informação estava escondida. No primeiro treino, o teste **não** confirmou esse efeito (notebook 03, seção 8).

## FID e KID

Métricas que comparam um conjunto de imagens geradas com um conjunto de imagens reais, usando as features de uma rede já treinada (Inception v3). Quanto menor, mais parecidos. O FID compara a média e a variação das features. O KID é mais confiável com poucas imagens. Detalhes na [ficha de métricas](../referencia/metricas.md).

## *Hub* (vizinho de todo mundo)

Em espaços com muitas dimensões, alguns pontos ficam perto de quase todos os outros e viram o "vizinho mais próximo" de muita coisa. No primeiro treino, a *Cadeira de Gauguin* foi a pintura mais próxima de quase todas as saídas, sem ter sido copiada. É uma limitação da distância em pixels.

## Tensor

Um jeito organizado de guardar números, com uma ou mais dimensões:

| Dimensões | Forma | Exemplo no projeto |
|---|---|---|
| 0 | um número | o valor de vermelho de um pixel (0 a 255) |
| 1 | uma lista | as 3 cores de um pixel (vermelho, verde, azul) |
| 2 | uma tabela | um canal de cor de uma imagem (128 x 128) |
| 3 | uma pilha de tabelas | uma imagem colorida (3 x 128 x 128: canais x altura x largura) |
| 4 | — | várias imagens juntas (N x 3 x 128 x 128) |

- **Indexação:** `x[0]` é a primeira imagem, `x[0][2]` é o canal azul dela e `x[0][2][10][5]` é o azul de um pixel.
- **`.shape`:** o tamanho de cada dimensão. Ex.: uma tabela de 2 linhas e 3 colunas tem shape `(2, 3)`.
- **Por que não usar uma lista do Python:** o tensor faz contas em todos os números de uma vez e pode rodar na GPU.

Onde aparece: em todo o projeto. Primeiro contato no [notebook 01](../../notebooks/01_exploracao.ipynb).
