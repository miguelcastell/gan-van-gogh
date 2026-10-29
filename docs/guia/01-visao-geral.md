# 1. Visão geral

## A ideia de uma GAN

Pense em dois personagens que treinam juntos:

- **Gerador (G):** um falsificador. Recebe um vetor de números aleatórios (o "ruído" **z**, com 100 números) e transforma isso numa imagem de 28x28.
- **Discriminador (D):** um perito. Recebe uma imagem e dá uma nota: "real" (veio do dataset) ou "falsa" (veio do G).

No treino, os dois se revezam:

1. O D vê imagens reais e falsas e aprende a separar umas das outras.
2. O G recebe o retorno do D ("essa imagem ainda parece falsa por causa disso") e se ajusta para enganá-lo melhor.

Se tudo der certo, o G fica tão bom que o D passa a acertar só na base do chute, uns 50%. Aí o G está gerando imagens parecidas com as reais.

## Por que é difícil

É um jogo entre dois modelos, não uma otimização simples:

- Se o D ficar bom rápido demais, o G não recebe um sinal útil e para de aprender.
- Se o G achar um "truque", por exemplo gerar só tênis porque engana bem, ele para de gerar as outras roupas. Isso se chama **colapso de modo**.

Boa parte do projeto é observar e medir esses problemas.

## O que o professor pede, traduzido

| Pedido do enunciado | Na prática |
|---|---|
| Escolher uma base simples | **Fashion-MNIST**: 70 mil fotos pequenas de roupas em 10 classes (camiseta, calça, tênis...) |
| Treinar uma GAN | **DCGAN**, a versão clássica com camadas convolucionais |
| Uma variação | **WGAN-GP**, que muda a regra do jogo para o treino ficar mais estável. Você compara as duas |
| Acompanhar G e D | Gráficos das losses e das notas do D ao longo do treino, mais uma grade de imagens por época |
| Avaliar | Realismo (FID), diversidade (gera todas as 10 classes?), memorização (copiou o treino?), exemplos ruins |
| Ética | Deepfakes, viés, privacidade, sempre ligando ao que você observou |

> As escolhas de dataset e modelos são sugestões, ainda não confirmadas com o professor. Veja a [ADR 0002](../adr/0002-dataset-e-modelos.md).

Próximo: [Roteiro das etapas](02-roteiro-etapas.md)
