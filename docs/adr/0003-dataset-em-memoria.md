# ADR 0003: Carregar o dataset inteiro em memória

- **Status:** Aceito
- **Data:** 2026-10-28

## Contexto

O `DataLoader` do PyTorch com `num_workers > 0` costuma dar erro no Windows, principalmente em notebooks. Com `num_workers=0`, ler e converter imagem por imagem deixa a GPU ociosa.

## Decisão

Carregar o Fashion-MNIST inteiro como um único tensor já normalizado em [-1, 1] e já na GPU (cerca de 190 MB), e gerar os minibatches embaralhando os índices (`iterate_minibatches` em `src/utils.py`).

## Alternativas consideradas

- **`DataLoader` com `num_workers=0`:** funciona, mas é mais lento.
- **`DataLoader` com workers e o guard `if __name__ == "__main__"`:** funciona em scripts, mas não em notebook e adiciona complexidade.

## Consequências

- Treino mais rápido e sem problemas de multiprocessamento no Windows.
- Não serve para bases grandes (ex.: CelebA em 64x64). Se a extensão opcional for feita, esta decisão precisa ser revista.
- Sem *data augmentation*, o que é aceitável para uma GAN neste caso.
