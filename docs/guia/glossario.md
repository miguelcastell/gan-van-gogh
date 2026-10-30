# Glossário

Termos explicados ao longo do projeto, em ordem alfabética. Ao aprender um termo novo, adicione aqui.

## Tensor

Um jeito organizado de guardar números, com uma ou mais dimensões:

| Dimensões | Forma | Exemplo no projeto |
|---|---|---|
| 0 | um número | o valor de vermelho de um pixel (0 a 255) |
| 1 | uma lista | as 3 cores de um pixel (vermelho, verde, azul) |
| 2 | uma tabela | um canal de cor de uma imagem (128 x 128) |
| 3 | uma pilha de tabelas | uma imagem colorida (3 x 128 x 128: canais x altura x largura) |
| 4 | — | várias imagens juntas (N x 3 x 128 x 128) |

- **Indexação:** `x[0]` é a primeira imagem, `x[0][2]` é o canal azul dela e `x[0][2][10][5]` é o azul de um pixel.
- **`.shape`:** o tamanho de cada dimensão. Ex.: uma tabela de 2 linhas e 3 colunas tem shape `(2, 3)`.
- **Por que não usar uma lista do Python:** o tensor faz contas em todos os números de uma vez e pode rodar na GPU.

Onde aparece: em todo o projeto. Primeiro contato no [notebook 01](../../notebooks/01_exploracao.ipynb).
