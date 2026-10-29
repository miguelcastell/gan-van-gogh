# 2. Roteiro: as etapas do projeto

Os prazos vêm do plano em [`projeto-final-gan.md`](../../projeto-final-gan.md).

## Etapa 1: Ambiente e dados (até 04/11)

- **O que você aprende:** como as imagens viram números. Cada pixel vira um valor entre -1 e 1.
- **Arquivo:** `notebooks/01_exploracao.ipynb`
- **O que observar:** como as roupas são, se as classes estão balanceadas.

## Etapa 2: DCGAN (até 11/11)

- **O que você aprende:** as arquiteturas do G e do D, as losses e o laço de treino.
- **Arquivos:** `src/models.py`, `src/train_dcgan.py`
- **O que observar:**
  - A grade de imagens melhorando época a época (o GIF em `outputs/dcgan/evolucao.gif`).
  - **D(x)**, a nota que o D dá às reais, e **D(G(z))**, a nota que ele dá às falsas, se aproximando de 0,5.

## Etapa 3: WGAN-GP (até 18/11)

- **O que você aprende:** por que a GAN original é instável e o que muda quando o D vira um "crítico" que dá uma pontuação em vez de uma probabilidade.
- **Arquivo:** `src/train_wgan_gp.py`
- **O que observar:** se as curvas ficam mais suaves e se as imagens saem melhores ou mais variadas.

## Etapa 4: Avaliação (até 25/11)

- **O que você aprende:** como medir algo subjetivo como "a imagem está boa?".
- **Arquivos:** `src/train_classifier.py`, `src/evaluate.py`, `src/plots.py`
- **O que observar:**
  - O **FID**: quanto menor, mais parecido com o real.
  - Um gráfico de barras com quantas imagens geradas caem em cada classe (diversidade e colapso de modo).
  - Pares "imagem gerada x imagem de treino mais parecida" (memorização).

## Etapa 5: Escrita (até 02/12)

- **O que você aprende:** a interpretar os resultados e conectá-los com a ética.
- **Arquivo:** `report/analise_etica.md`
- O texto deve se basear nas suas figuras, não em teoria genérica.

Anterior: [Visão geral](01-visao-geral.md) · Próximo: [Próximos passos](03-proximos-passos.md)
