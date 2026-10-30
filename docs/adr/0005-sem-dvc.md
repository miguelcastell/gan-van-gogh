# ADR 0005: Não usar DVC para versionar dados e modelos

- **Status:** Aceito
- **Data:** 2026-10-28

## Contexto

O DVC (*Data Version Control*) versiona arquivos grandes (datasets, modelos) fora do git, num armazenamento remoto. A dúvida era se ele traria ganho de reprodutibilidade para este projeto.

Situação atual:
- O `vangogh2photo` é público e fixo, e qualquer pessoa baixa a mesma versão.
- Cada checkpoint da CycleGAN guarda 2 geradores e 2 discriminadores. Eles são grandes, mas só o último e alguns intermediários precisam ser guardados.
- Logs, métricas e figuras são pequenos e já vão no git.

## Decisão

Não usar DVC. A reprodutibilidade fica garantida por:
- download do dataset por código, sempre da mesma fonte;
- seeds fixas (`--seed`) e hiperparâmetros salvos em cada checkpoint;
- logs, métricas e figuras versionados no git;
- registro em `docs/experimentos/`.

Se for preciso distribuir o gerador treinado, o checkpoint final vai anexado a uma GitHub Release (`gh release create`). Para a demonstração basta o gerador foto → Van Gogh.

## Alternativas consideradas

- **DVC com remoto no Google Drive:** hoje exige criar credenciais próprias no Google Cloud. É mais uma ferramenta para aprender em pouco tempo, e o ganho é pequeno com dados públicos.
- **Git LFS para os checkpoints:** a cota gratuita do GitHub é limitada, e os checkpoints intermediários não precisam ser guardados.
- **Versionar checkpoints direto no git:** deixaria o repositório pesado com arquivos binários que mudam a cada treino.

## Consequências

- Uma ferramenta a menos para configurar e aprender. O tempo vai para o projeto em si.
- Os checkpoints intermediários ficam só na máquina local. Como o treino leva horas, vale fazer backup manual do checkpoint final.
- Reavaliar se surgir um dataset próprio (por exemplo, fotos tiradas para a demonstração que precisem ser compartilhadas).
