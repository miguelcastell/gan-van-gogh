# 1. Visão geral

## O que o projeto faz

Você tira uma foto, por exemplo de uma estátua, e recebe a mesma foto com **o céu pintado como Van Gogh pintaria**. O resto da foto fica igual.

```
foto → [segmentador pré-treinado] → máscara do céu
foto → [gerador da CycleGAN foto → Van Gogh] → foto toda estilizada
resultado = estilizada onde é céu + original no resto
```

A GAN é a parte do meio, e é ela que você treina. O segmentador é um modelo pronto que só diz "aqui é céu, aqui não é".

## A ideia de uma GAN

Pense em dois personagens que treinam juntos:

- **Gerador (G):** um falsificador. Cria imagens tentando fazer com que pareçam reais.
- **Discriminador (D):** um perito. Recebe uma imagem e dá uma nota: "real" ou "falsa" (veio do G).

No treino, os dois se revezam:

1. O D vê imagens reais e falsas e aprende a separar umas das outras.
2. O G recebe o retorno do D ("essa imagem ainda parece falsa por causa disso") e se ajusta para enganá-lo melhor.

Se tudo der certo, o G fica tão bom que o D não consegue mais separar o que é real do que é falso.

## A CycleGAN: dois falsificadores e dois peritos

Numa GAN comum, o gerador parte de números aleatórios. Na CycleGAN ele parte de uma **imagem**: recebe uma foto e devolve uma pintura.

São dois pares treinando ao mesmo tempo:

| | Converte | Quem julga |
|---|---|---|
| **G** | foto → pintura | **D_pintura**: "isso parece um Van Gogh de verdade?" |
| **F** | pintura → foto | **D_foto**: "isso parece uma foto de verdade?" |

### O problema que a *cycle loss* resolve

Só com o D_pintura, o G poderia trapacear: receber a foto da estátua e devolver uma pintura bonita de girassóis. O D_pintura aprovaria, porque parece Van Gogh, mas a estátua sumiu.

A solução é o **ciclo**: converter a foto em pintura e de volta em foto, `F(G(foto))`, tem que dar a foto original. Se o G tivesse jogado a estátua fora, o F não teria como recuperá-la. Esse erro de ida e volta é a ***cycle loss***, e ela obriga o G a mudar só o **estilo**, preservando o **conteúdo**.

Por isso a CycleGAN **não precisa de pares** (a foto exata de cada cena que Van Gogh pintou), que não existem.

## Por que é difícil

É um jogo entre modelos, não uma otimização simples:

- Se o D ficar bom rápido demais, o G não recebe um sinal útil e para de aprender.
- O G pode achar um "truque", como aplicar sempre a mesma textura em qualquer foto, ignorando o que há nela.
- Há só umas centenas de pinturas de Van Gogh, contra milhares de fotos. O D_pintura pode acabar decorando as pinturas.

Boa parte do projeto é observar e medir esses problemas.

## Uma limitação para ter clara desde já

A CycleGAN aprende o **estilo geral** de Van Gogh a partir de todas as pinturas: as pinceladas, as cores, a textura. Ela **não** aprende a *Noite Estrelada* em si. O céu vai sair com cara de Van Gogh, mas sem os redemoinhos exatos e a lua daquele quadro.

## O que o professor pede, traduzido

| Pedido do enunciado | Na prática |
|---|---|
| Escolher uma base simples | **`vangogh2photo`**: umas 400 pinturas de Van Gogh e uns 6 mil fotos de paisagens |
| Treinar uma GAN | **CycleGAN** foto ↔ Van Gogh, em 128x128 |
| Uma variação | A CycleGAN **sem a *cycle loss***, para ver o que acontece sem ela |
| Acompanhar G e D | Gráficos das losses e uma grade com as mesmas fotos convertidas a cada época |
| Avaliar | Estilo (FID/KID), conteúdo preservado (ida e volta), memorização (copiou as pinturas?), exemplos ruins |
| Ética | Imitação de estilo, autoria, manipulação de fotos, sempre ligando ao que você observou |

As decisões estão na [ADR 0002](../adr/0002-dataset-e-modelos.md).

Próximo: [Roteiro das etapas](02-roteiro-etapas.md)
