# 5. KDD e a exploração dos dados

## Onde fica o dataset

O `vangogh2photo` foi publicado junto com a CycleGAN. Depois de baixado, ele fica em:

```
data/
└── vangogh2photo/
    ├── trainA/   ← pinturas de Van Gogh (treino)
    ├── testA/    ← pinturas de Van Gogh (teste)
    ├── trainB/   ← fotos de paisagens (treino)
    └── testB/    ← fotos de paisagens (teste)
```

São arquivos JPEG coloridos de 256x256 (RGB). "A" e "B" são só os nomes dos dois domínios: A = Van Gogh, B = foto.

| Pasta | Imagens |
|---|---|
| `trainA` | 400 pinturas |
| `testA` | 400 pinturas, **as mesmas do `trainA`** |
| `trainB` | 6.287 fotos |
| `testB` | 751 fotos |

O `testA` ser uma cópia do `trainA` foi descoberto no notebook 01 e muda a avaliação (veja a [ADR 0002](../adr/0002-dataset-e-modelos.md)). Os detalhes completos (duplicatas, uso de cada conjunto, pré-processamento) estão na [ficha dos dados](../referencia/dados.md).

A pasta `data/` está no `.gitignore`: o dataset é público e qualquer pessoa baixa da mesma fonte, então não vai para o GitHub.

## O que é KDD

**KDD** (*Knowledge Discovery in Databases*, ou Descoberta de Conhecimento em Bases de Dados) é um processo em 5 fases para extrair conhecimento de dados. O projeto passa por todas elas:

| Fase do KDD | No projeto | Onde |
|---|---|---|
| 1. Seleção | Escolher o `vangogh2photo` e separar treino e teste | Notebook 01 |
| 2. Pré-processamento | Conferir a qualidade: tamanhos, imagens corrompidas, duplicatas, desequilíbrio entre domínios | Notebook 01 |
| 3. Transformação | Redimensionar para 128x128, converter os pixels de 0–255 para [-1, 1] e montar os tensores | Notebook 01, `src/utils.py` |
| 4. Mineração (modelagem) | Treinar a CycleGAN completa e a versão sem *cycle loss* | `src/train_cyclegan.py` |
| 5. Interpretação e avaliação | FID/KID, preservação do conteúdo, memorização, análise ética | `src/evaluate.py`, `report/` |

Para uma disciplina de Aprendizado Não Supervisionado, organizar o relatório nessas fases deixa claro o caminho percorrido. A proposta está registrada na [ADR 0004](../adr/0004-kdd-como-estrutura.md).

## Onde não vale exagerar

Com imagens, boa parte da análise exploratória clássica não diz nada. Não há valores faltando, nem colunas para correlacionar. Gráfico só para preencher espaço não ajuda: cada análise precisa alimentar uma etapa posterior.

## As análises que valem a pena (e por quê)

| Análise | Para que serve depois |
|---|---|
| Quantidade de imagens em cada domínio | Mostra o desequilíbrio (poucas pinturas, muitas fotos), que explica o risco de o D_pintura decorar o treino |
| Histograma de cores (pinturas x fotos) | Mostra em números o que o gerador precisa mudar: Van Gogh usa mais amarelo e azul saturados |
| Imagem média de cada domínio | Mostra a composição típica: fotos de paisagem têm o céu em cima, e isso importa para a máscara |
| Projeção 2D (PCA, t-SNE) | Achata as imagens em 2 dimensões para ver se pinturas e fotos formam grupos separados. É aprendizado não supervisionado, e na Etapa 4 dá para projetar as saídas da GAN no mesmo espaço e ver de que lado caem |
| Duplicatas teste → treino | Mostrou que as pinturas de teste são as mesmas do treino. Por isso a referência de memorização passou a ser a distância entre pinturas diferentes |
| Origem dos dados | Pinturas de um único artista, coletadas do WikiArt; fotos de paisagens do Flickr. Material para a seção de viés |

## Como praticar

O notebook [`notebooks/01_exploracao.ipynb`](../../notebooks/01_exploracao.ipynb) cobre as fases 1 a 3 no formato de exercícios:

- cada `TODO` tem um `___` para completar e uma dica;
- uma célula de verificação mostra ✅ ou ❌;
- as perguntas ✍️ pedem interpretação, que é o que vai para o relatório.

No final, a tabela de resumo do notebook deve virar um registro em [`experimentos/`](../experimentos/).

Anterior: [Como aprender](04-como-aprender.md) · Próximo: [A CycleGAN por dentro](06-cyclegan-por-dentro.md)
