# ADR 0005: Não usar DVC para versionar dados e modelos

- **Status:** Aceito
- **Data:** 2026-10-28

## Contexto

O DVC (*Data Version Control*) versiona arquivos grandes (datasets, modelos) fora do git, num armazenamento remoto. A dúvida era se ele traria ganho de reprodutibilidade para este projeto.

Situação atual:
- O Fashion-MNIST é público e fixo, e o torchvision baixa sempre a mesma versão.
- Os checkpoints têm cerca de 20–25 MB cada (cerca de 200 MB por treino completo), mas um treino leva minutos e pode ser refeito com seed fixa.
- Logs, métricas e figuras são pequenos e já vão no git.

## Decisão

Não usar DVC. A reprodutibilidade fica garantida por:
- download automático do dataset;
- seeds fixas (`--seed`) e hiperparâmetros salvos em cada checkpoint;
- logs, métricas e figuras versionados no git;
- registro em `docs/experimentos/`.

Se for preciso distribuir o gerador treinado, o checkpoint final vai anexado a uma GitHub Release (`gh release create`).

## Alternativas consideradas

- **DVC com remoto no Google Drive:** hoje exige criar credenciais próprias no Google Cloud. É mais uma ferramenta para aprender em pouco tempo, e o ganho é pequeno com dados públicos e treinos curtos.
- **Git LFS para os checkpoints:** a cota gratuita do GitHub é limitada, e os checkpoints intermediários não precisam ser guardados.
- **Versionar checkpoints direto no git:** deixaria o repositório pesado com arquivos binários que mudam a cada treino.

## Consequências

- Uma ferramenta a menos para configurar e aprender. O tempo vai para o projeto em si.
- Os checkpoints intermediários ficam só na máquina local. Se forem perdidos, o treino precisa ser refeito.
- Reavaliar se for feita a extensão em 64x64 (treinos longos) ou se surgir um dataset próprio.
