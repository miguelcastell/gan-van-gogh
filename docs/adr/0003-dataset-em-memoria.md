# ADR 0003: Carregar o dataset inteiro em memória

- **Status:** Proposto
- **Data:** 2026-10-29

## Contexto

O `DataLoader` do PyTorch com `num_workers > 0` costuma dar erro no Windows, principalmente em notebooks. Com `num_workers=0`, abrir e redimensionar um JPEG a cada passo deixa a GPU ociosa.

O `vangogh2photo` tem 7.838 imagens. Redimensionadas para 128x128 com 3 canais em `uint8`, ocupam uns 385 MB.

## Decisão

Ler todas as imagens uma vez, redimensionar para 128x128 e guardar como um tensor `uint8` (cerca de 385 MB). A conversão para `float` em [-1, 1] é feita só no batch, na hora do treino, e os minibatches são sorteados por índice. O tensor redimensionado fica salvo em `data/vangogh2photo_128/`, para não reler os JPEGs a cada treino (`load_domain` em `src/utils.py`).

## Alternativas consideradas

- **`DataLoader` com `num_workers=0` lendo do disco:** funciona, mas é mais lento.
- **`DataLoader` com workers e o guard `if __name__ == "__main__"`:** funciona em scripts, mas não em notebook e adiciona complexidade.
- **Guardar tudo já em `float32`:** seria 4 vezes maior (cerca de 1,5 GB), à toa.

## Consequências

- Treino mais rápido e sem problemas de multiprocessamento no Windows.
- Redimensionar uma vez fixa a resolução. Se for usado 256x256, o tensor vai para cerca de 1,3 GB em `uint8`, o que ainda cabe na RAM.
- *Data augmentation* (o espelhamento horizontal usado pela CycleGAN) é feito no batch, sem custo extra de disco.
