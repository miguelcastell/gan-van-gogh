# 5. KDD e a exploração dos dados

## Onde fica o dataset

Não é preciso baixar nada à mão. Em `src/utils.py`, a função `load_fashion_mnist` chama:

```python
datasets.FashionMNIST(DATA_DIR, train=train, download=True)
```

`DATA_DIR` aponta para a pasta `data/`. Na primeira execução (do notebook ou do treino), o torchvision baixa cerca de 30 MB e salva em:

```
data/
└── FashionMNIST/
    └── raw/
        ├── train-images-idx3-ubyte   ← 60 mil imagens de treino
        ├── train-labels-idx1-ubyte   ← o rótulo (classe) de cada uma
        ├── t10k-images-idx3-ubyte    ← 10 mil imagens de teste
        └── t10k-labels-idx1-ubyte
```

Nas execuções seguintes ele só lê o que já está lá. A pasta `data/` está no `.gitignore`: o dataset é público e qualquer pessoa baixa com o mesmo código, então não vai para o GitHub.

## O que é KDD

**KDD** (*Knowledge Discovery in Databases*, ou Descoberta de Conhecimento em Bases de Dados) é um processo em 5 fases para extrair conhecimento de dados. O projeto passa por todas elas:

| Fase do KDD | No projeto | Onde |
|---|---|---|
| 1. Seleção | Escolher o Fashion-MNIST e separar treino e teste | Notebook 01 |
| 2. Pré-processamento | Conferir a qualidade: faixa de valores, balanceamento, duplicatas | Notebook 01 |
| 3. Transformação | Converter os pixels de 0–255 para [-1, 1] e montar os tensores | Notebook 01, `src/utils.py` |
| 4. Mineração (modelagem) | Treinar a DCGAN e a WGAN-GP | `src/train_*.py` |
| 5. Interpretação e avaliação | FID, diversidade, memorização, análise ética | `src/evaluate.py`, `report/` |

Para uma disciplina de Aprendizado Não Supervisionado, organizar o relatório nessas fases deixa claro o caminho percorrido. A proposta está registrada na [ADR 0004](../adr/0004-kdd-como-estrutura.md).

## Onde não vale exagerar

Com imagens, boa parte da análise exploratória clássica não diz nada. Não há valores faltando, nem colunas para correlacionar, e o Fashion-MNIST já vem limpo e balanceado. Gráfico só para preencher espaço não ajuda: cada análise precisa alimentar uma etapa posterior.

## As análises que valem a pena (e por quê)

| Análise | Para que serve depois |
|---|---|
| Contagem por classe | É o padrão de comparação da diversidade: com 6.000 imagens por classe, uma GAN perfeita geraria cerca de 10% de cada uma (Etapa 4) |
| Imagem média e desvio padrão por classe | Mostram o formato típico e a variação de cada roupa. Classes muito variadas tendem a ser mais difíceis para o gerador |
| Distância entre classes | Camiseta, camisa, pulôver e casaco se parecem. A previsão é que a GAN e o classificador errem ali, e isso é conferido na Etapa 4 |
| Projeção 2D (PCA, t-SNE) | Achata as 784 dimensões em 2 para ver como as classes se agrupam. É aprendizado não supervisionado, e na Etapa 4 dá para projetar as imagens geradas no mesmo espaço |
| Quase-duplicatas teste → treino | É a referência do teste de memorização: mostra a distância "normal" entre uma imagem nova e o treino |
| Origem dos dados | Fotos de catálogo da Zalando (fundo limpo, roupa centralizada, moda ocidental). Material para a seção de viés |

## Como praticar

O notebook [`notebooks/01_exploracao.ipynb`](../../notebooks/01_exploracao.ipynb) cobre as fases 1 a 3 e todas as análises acima, no formato de exercícios:

- cada `TODO` tem um `___` para completar e uma dica;
- uma célula de verificação mostra ✅ ou ❌;
- as perguntas ✍️ pedem interpretação, que é o que vai para o relatório.

No final, a tabela de resumo do notebook deve virar um registro em [`experimentos/`](../experimentos/).

Anterior: [Como aprender](04-como-aprender.md)
