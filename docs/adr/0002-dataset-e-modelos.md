# ADR 0002: Fashion-MNIST com DCGAN e WGAN-GP

- **Status:** Proposto (aguardando confirmação do professor)
- **Data:** 2026-10-28

## Contexto

O enunciado pede uma base simples, uma GAN básica e uma variação para comparar. O treino é 100% local numa RTX 3050 de 6 GB, com prazo em 02/12/2026.

## Decisão

- **Dataset:** Fashion-MNIST (28x28, escala de cinza, 10 classes).
- **Modelo base:** DCGAN.
- **Variação:** WGAN-GP, com o mesmo gerador, para que a comparação isole o efeito da loss e do crítico.

## Alternativas consideradas

- **MNIST:** simples demais; a comparação entre modelos fica pouco interessante.
- **CelebA / CIFAR-10 em 64x64:** mais caro de treinar em 6 GB; fica como extensão opcional.
- **DCGAN com spectral norm** no lugar da WGAN-GP: válida, mas a WGAN-GP tem motivação teórica mais fácil de explicar no relatório.

## Consequências

- Treinos rápidos (minutos por modelo), o que permite repetir experimentos.
- As 10 classes permitem medir diversidade e colapso de modo com um classificador.
- Imagens de roupas não envolvem pessoas: a discussão ética sobre rostos e deepfakes será por analogia, não por evidência direta.
