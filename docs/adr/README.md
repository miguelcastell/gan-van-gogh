# ADRs: registro de decisões

Um **ADR** (*Architecture Decision Record*) é um documento curto que registra **uma** decisão técnica: o contexto, o que foi decidido, as alternativas e as consequências. Serve para você (e o professor) entenderem depois *por que* o projeto é como é.

## Quando escrever um

- Ao escolher entre alternativas reais (ex.: com ou sem a identity loss).
- Ao mudar uma decisão anterior. Não apague o ADR antigo: marque como `Substituído por ADR-XXXX` e crie um novo.
- Não precisa de ADR para detalhes triviais ou fáceis de desfazer.

## Como criar

1. Copie [`template.md`](template.md) para `NNNN-titulo-curto.md`, com o próximo número.
2. Preencha e adicione na tabela abaixo.

## Status possíveis

`Proposto` (aguardando confirmação) · `Aceito` · `Rejeitado` · `Substituído por ADR-NNNN`

## Índice

| Nº | Título | Status | Data |
|---|---|---|---|
| [0001](0001-registrar-decisoes.md) | Registrar decisões com ADRs | Aceito | 2026-10-28 |
| [0002](0002-dataset-e-modelos.md) | CycleGAN no vangogh2photo, com o estilo aplicado só no céu | Aceito | 2026-10-29 |
| [0003](0003-dataset-em-memoria.md) | Carregar o dataset inteiro em memória | Proposto | 2026-10-29 |
| [0004](0004-kdd-como-estrutura.md) | Usar o KDD para organizar o projeto e o relatório | Proposto | 2026-10-28 |
| [0005](0005-sem-dvc.md) | Não usar DVC para versionar dados e modelos | Aceito | 2026-10-28 |
| [0006](0006-notebooks-e-scripts.md) | Notebooks guiados para aprender, scripts para o treino longo | Aceito | 2026-10-29 |
| [0007](0007-rastreamento-sem-mlflow.md) | Rastrear os experimentos com arquivos e git, sem MLflow | Proposto | 2026-10-29 |
