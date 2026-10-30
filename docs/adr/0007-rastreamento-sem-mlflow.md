# ADR 0007: Rastrear os experimentos com arquivos e git, sem MLflow

- **Status:** Proposto
- **Data:** 2026-10-29

## Contexto

O autor quer que cada treino fique bem registrado, para servir de base a artigos. Ferramentas como o **MLflow** são feitas para isso: guardam os parâmetros, as métricas e os arquivos de cada execução e mostram tudo numa interface web.

Neste projeto os treinos são poucos (alguns por semana, de horas cada), rodam numa máquina só, e o código já grava:
- os **parâmetros** de cada treino dentro do checkpoint (`args`);
- as **losses** a cada 100 passos em `outputs/<experimento>/logs.csv`;
- as **métricas finais** em `outputs/<experimento>/metrics.json`;
- as **grades por época** e o GIF em `outputs/<experimento>/samples/`.

## Decisão

Não usar MLflow por enquanto. Cada treino é rastreado por:
1. os arquivos acima, versionados no git (menos os checkpoints, ver [ADR 0005](0005-sem-dvc.md));
2. um registro em `docs/experimentos/` com o commit, o comando, a configuração, os dados, a previsão, um diário da execução e os resultados;
3. as fichas de [`docs/referencia/`](../referencia/) com os padrões.

## Alternativas consideradas

- **MLflow:** compara execuções numa interface, mas é mais uma ferramenta para instalar e aprender, e o ganho é pequeno com poucos treinos. Os registros em texto também vão direto para o artigo, o que uma interface web não faz.
- **TensorBoard:** bom para ver as curvas ao vivo durante o treino. Pode valer a pena se os treinos ficarem frequentes, e é simples de adicionar depois.
- **Weights & Biases:** serviço externo, que exige conta e manda os dados para fora da máquina.

## Consequências

- Nada novo para instalar, e tudo fica legível no GitHub.
- A comparação entre execuções é manual: a tabela gerada por `python -m src.plots` e os registros de experimento.
- Reavaliar se o número de treinos crescer (por exemplo, uma busca de hiperparâmetros).
