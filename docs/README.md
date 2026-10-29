# Documentação do projeto

Índice de toda a documentação. O plano original do projeto está em [`../projeto-final-gan.md`](../projeto-final-gan.md).

## Estrutura

| Pasta | Para que serve | Quando adicionar algo |
|---|---|---|
| [`guia/`](guia/) | Material de estudo: conceitos e roteiro do projeto | Quando aprender algo que vale explicar com as próprias palavras |
| [`adr/`](adr/) | *Architecture Decision Records*: decisões técnicas e o porquê delas | Sempre que escolher entre alternativas (modelo, métrica, hiperparâmetro...) |
| [`boas-praticas/`](boas-praticas/) | Convenções do projeto (código, git, nomes, reprodutibilidade) | Quando definir uma regra que deve valer para todo o projeto |
| [`experimentos/`](experimentos/) | Diário de experimentos: o que rodou, o que esperava, o que aconteceu | A cada treino ou teste relevante |

## Guia de estudo

1. [Visão geral: o que é uma GAN e o que o projeto pede](guia/01-visao-geral.md)
2. [Roteiro: as etapas do projeto](guia/02-roteiro-etapas.md)
3. [Próximos passos: o que fazer agora](guia/03-proximos-passos.md)
4. [Como aprender (e não só rodar)](guia/04-como-aprender.md)
5. [KDD e a exploração dos dados](guia/05-kdd.md)
6. [DCGAN x WGAN-GP: a configuração, linha por linha](guia/06-dcgan-vs-wgan-gp.md)

## Convenções desta pasta

- Arquivos em minúsculas, com hífen: `01-visao-geral.md`.
- Documentos com ordem de leitura ou numeração levam prefixo numérico (`01-`, `0001-`).
- Cada subpasta tem um `README.md` com índice e, quando fizer sentido, um `template.md`.
- Ao criar um documento novo, adicione o link no `README.md` da subpasta.
- As regras de quando e onde documentar estão na skill [`.claude/skills/documentacao/SKILL.md`](../.claude/skills/documentacao/SKILL.md), que o Claude segue automaticamente. Se mudar as convenções aqui, atualize a skill também.
