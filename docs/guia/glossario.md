# Glossário

Termos explicados ao longo do projeto, em ordem alfabética. Ao aprender um termo novo, adicione aqui.

## Tensor

Um jeito organizado de guardar números, com uma ou mais dimensões:

| Dimensões | Forma | Exemplo no projeto |
|---|---|---|
| 0 | um número | um pixel (0 a 255) |
| 1 | uma lista | uma linha de uma imagem (28 números) |
| 2 | uma tabela | uma imagem (28 x 28) |
| 3 | uma pilha de tabelas | o dataset de treino (60.000 x 28 x 28) |
| 4 | — | um batch no treino (128 x 1 x 28 x 28: imagens x canais x altura x largura) |

- **Indexação:** `x_train[0]` é a primeira imagem, `x_train[0][5]` é a linha 5 dela e `x_train[0][5][10]` é um pixel.
- **`.shape`:** o tamanho de cada dimensão. Ex.: uma tabela de 2 linhas e 3 colunas tem shape `(2, 3)`.
- **Por que não usar uma lista do Python:** o tensor faz contas em todos os números de uma vez e pode rodar na GPU.

Onde aparece: em todo o projeto. Primeiro contato no TODO 1 do [notebook 01](../../notebooks/01_exploracao.ipynb).
