# ADR 0001: Registrar decisões com ADRs

- **Status:** Aceito
- **Data:** 2026-10-28

## Contexto

O projeto envolve várias escolhas (dataset, modelos, métricas, hiperparâmetros) e o relatório final precisa justificá-las. Parte do plano inicial foi sugerida por IA, então é importante deixar claro o que foi decidido conscientemente e por quê.

## Decisão

Registrar cada decisão relevante como um ADR curto em `docs/adr/`, seguindo o [`template.md`](template.md).

## Alternativas consideradas

- **Não documentar:** mais rápido, mas as justificativas se perdem até a hora de escrever o relatório.
- **Um único documento de decisões:** fica longo e difícil de manter; mudanças de decisão ficam sem histórico.

## Consequências

- As justificativas do relatório já ficam escritas ao longo do projeto.
- Exige alguns minutos de disciplina a cada decisão.
