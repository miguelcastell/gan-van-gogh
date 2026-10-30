# Referência técnica

Fichas com os fatos exatos do projeto: dados, hiperparâmetros e métricas. Servem de fonte para a metodologia do artigo e do relatório.

A diferença para as outras pastas: o **ADR** explica *por que* algo foi decidido, o **experimento** conta o que aconteceu num treino, e a **referência** diz *o que é*, com os números exatos do momento.

## Regras

- Só números conferidos (no código, nos dados ou num experimento), com a data e o commit em que foram conferidos.
- Sempre que um padrão do código, os dados ou o ambiente mudarem, atualize a ficha correspondente no mesmo commit.
- Valores de um treino específico ficam no registro do experimento, não aqui. Aqui ficam os padrões.

## Índice

| Ficha | Conteúdo |
|---|---|
| [dados.md](dados.md) | Origem, composição, duplicatas, uso de cada conjunto por etapa, pré-processamento e limitações |
| [hiperparametros.md](hiperparametros.md) | Treino, losses, arquitetura, avaliação, máscara do céu e versões do ambiente |
| [metricas.md](metricas.md) | Colunas do `logs.csv`, campos do `metrics.json`, FID/KID e figuras geradas |
