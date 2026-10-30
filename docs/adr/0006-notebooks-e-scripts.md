# ADR 0006: Notebooks guiados para aprender, scripts para o treino longo

- **Status:** Aceito
- **Data:** 2026-10-29

## Contexto

O autor do projeto quer fazer o trabalho pelo notebook, para entender cada parte. Mas o treino completo da CycleGAN leva horas (cerca de 130 ms por passo, ver o [experimento de tempo](../experimentos/2026-10-29-tempo-por-passo.md)). Um treino longo dentro do Jupyter trava se o computador dormir ou se a aba fechar, e não continua de onde parou.

## Decisão

- **Notebooks (`notebooks/`):** guiados, com `TODO`s e verificações ✅/❌. O autor explora os dados, monta o gerador, o discriminador e as losses peça por peça e roda treinos curtos (centenas de passos).
- **Scripts (`src/`):** o mesmo código, completo. Servem de **gabarito** para conferir os `TODO`s e rodam o treino longo no terminal, com checkpoints e `--resume`.
- O notebook 02 confere que o gerador montado nele tem a mesma arquitetura do `src/models.py` (mesmo número de parâmetros).

## Alternativas consideradas

- **Tudo no notebook, inclusive o treino longo:** um único lugar, mas arrisca perder horas de treino e não tem como continuar de onde parou.
- **Tudo em scripts:** mais robusto, mas o autor só leria código pronto, o que ensina menos.
- **Apagar o `src/` e o autor escrever do zero:** ensina mais, mas sem gabarito não há como conferir, e o prazo fica apertado.

## Consequências

- Cada conceito é praticado no notebook antes de ser usado no treino real.
- O código existe em dois lugares. Se a arquitetura mudar no `src/`, o notebook 02 precisa mudar junto (a verificação de parâmetros acusa a diferença).
- O gabarito está à mão, e cabe ao autor tentar antes de olhar.
