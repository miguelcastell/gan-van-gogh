# ADR 0004: Usar o KDD para organizar o projeto e o relatório

- **Status:** Proposto
- **Data:** 2026-10-28

## Contexto

O relatório precisa mostrar o caminho do dado bruto até a avaliação dos modelos. A disciplina é de Aprendizado Não Supervisionado, em que o processo KDD é uma referência conhecida. Com imagens já limpas e balanceadas, uma análise exploratória extensa teria pouco valor.

## Decisão

Organizar o projeto e o relatório nas 5 fases do KDD (seleção, pré-processamento, transformação, mineração, interpretação). A análise exploratória se limita ao que alimenta etapas posteriores. Ver [guia 05](../guia/05-kdd.md).

## Alternativas consideradas

- **Seguir só as etapas do plano (Etapa 1 a 5):** funciona, mas não conecta o trabalho a um processo conhecido da disciplina.
- **CRISP-DM:** também é válido, mas tem fases de negócio (entendimento do negócio, implantação) que não se aplicam a um trabalho acadêmico.
- **Análise exploratória completa:** gráficos sem uso posterior ocupam espaço sem acrescentar nada.

## Consequências

- O relatório ganha uma estrutura reconhecível, e cada análise tem um porquê.
- A fase de "mineração", no caso de uma GAN, é modelagem generativa, não descoberta de padrões no sentido clássico. Vale comentar essa diferença no relatório.
- Depende de o professor aceitar essa estrutura (confirmar junto com o formato da entrega).
