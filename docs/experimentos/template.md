# Experimento: nome curto

- **Data:** AAAA-MM-DD
- **Commit do código:** `git log -1 --format=%h`
- **Comando:** `python -m src.train_cyclegan --epochs 1 --out ...`
- **Pasta de saída:** `outputs/...`

## Pergunta

O que eu quero descobrir com este experimento?

## Previsão (antes de rodar)

O que espero ver nas imagens e nas curvas? Por quê?

## Configuração

Só o que difere dos padrões da [ficha de hiperparâmetros](../referencia/hiperparametros.md). O checkpoint guarda todos os valores em `args`.

| Parâmetro | Valor |
|---|---|
| Épocas / fotos por época | |
| λ_cycle / λ_identity | |
| Seed | |
| Outros | |

## Dados

Quais conjuntos e quantas imagens ([ficha dos dados](../referencia/dados.md)). Ex.: 400 pinturas de `trainA` e 2.000 fotos de `trainB` sorteadas com a seed 42. Avaliação nas 751 fotos de `testB`.

## Resultado

- Tempo de treino (total e por passo):
- Métricas (copie do `metrics.json`):
- Imagens / figuras: (link para os arquivos)

## O que observei

Bateu com a previsão? O que surpreendeu?

## Conclusão e próximos passos

O que aprendi e o que vou testar a seguir.
