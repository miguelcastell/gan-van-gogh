# Ficha dos dados

Tudo o que é preciso saber sobre os dados para descrevê-los no artigo. Números conferidos em 2026-10-29 com o dataset baixado e o código no commit `fed4c39`.

## Origem

| | |
|---|---|
| Dataset | `vangogh2photo`, publicado com a CycleGAN (Zhu et al., 2017) |
| Fonte do download | http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip (`python -m src.download_data`) |
| Pinturas | Obras de Van Gogh coletadas do WikiArt pelos autores da CycleGAN. Domínio público |
| Fotos | Fotos de paisagens coletadas do Flickr pelos autores da CycleGAN. Licença individual de cada foto não informada |
| Pares | Não há: nenhuma foto mostra a cena de uma pintura |

## Composição

| Pasta | Domínio | Imagens | Tamanho em disco | Formato |
|---|---|---|---|---|
| `trainA` | Pinturas | 400 | 17,6 MB | JPEG 256x256 RGB |
| `testA` | Pinturas | 400, **idênticas às de `trainA`** | 17,6 MB | JPEG 256x256 RGB |
| `trainB` | Fotos | 6.287 | 235,0 MB | JPEG 256x256 RGB |
| `testB` | Fotos | 751 | 24,6 MB | JPEG 256x256 RGB |

- **Pinturas únicas:** 400. Os nomes vão de `00001.jpg` a `00755.jpg`, com lacunas.
- **Fotos:** 7.038 no total. Os nomes são datas de 2013-11-08 a 2017-01-03 (treino) e de 2014-08-01 a 2015-04-30 (teste).
- **Proporção no treino:** 15,7 fotos por pintura.
- **Todas as 7.838 imagens** têm o mesmo tamanho (256x256) e o mesmo modo (RGB). Nenhuma está em escala de cinza ou corrompida.

## Qualidade: duplicatas

Medido com a distância L2 entre as imagens em 128x128, nos valores de pixel originais (0 a 255), com a conta exata do `torch.cdist` (`compute_mode="donot_use_mm_for_euclid_dist"`). O modo padrão erra com imagens quase iguais (veja o TODO 4 do notebook 01). Critério: distância < 1 = cópia exata.

| Comparação | Resultado |
|---|---|
| `testA` → `trainA` | **400 de 400 são cópias.** Não existem pinturas de teste separadas |
| `testB` → `trainB` | 0 cópias. A menor distância passa de 200, e a mediana é 10.065. O teste de fotos é limpo |
| Dentro de `trainA` | 4 pinturas têm uma cópia exata entre as outras (2 pares) |
| Dentro de `trainB` | 17 fotos têm uma cópia exata entre as outras, e 23 têm uma quase cópia (distância < 200) |
| Dentro de `testB` | 2 fotos têm uma cópia exata (1 par) |

Para comparação, a mediana da distância até a imagem mais parecida é de cerca de 10.000 a 12.000 em todos os conjuntos. As duplicatas internas são poucas e não foram removidas.

## Quais dados cada etapa usa

| Etapa | Pinturas | Fotos | Observações |
|---|---|---|---|
| Treino (`src/train_cyclegan.py`) | `trainA`, as 400 | `trainB`. Todas, ou um subconjunto de `--photos N` | O subconjunto é sorteado com a seed do treino (`--seed`, padrão 42), então se repete com a mesma seed |
| Grade de cada época | — | As 8 primeiras de `testB`, em ordem alfabética | Sempre as mesmas, para comparar as épocas |
| Avaliação (`src/evaluate.py`) | `trainA`, as 400, como referência de estilo e de memorização | `testB`, as 751, que o modelo nunca viu | — |
| Notebook 01 | Todas | Todas | O PCA usa as 400 primeiras de cada domínio |
| Notebook 02 | `trainA` | `trainB` (todas) | Treino curto de 300 passos. A grade usa as 8 primeiras de `testB` |
| Demonstração (`src/stylize.py`) | — | Fotos próprias | Fora do dataset |

## Pré-processamento

1. **Redimensionar:** 256x256 → 128x128, com interpolação bicúbica (`load_image` em `src/utils.py`).
2. **Guardar:** tensor `uint8` com formato (N, 3, 128, 128), salvo em `data/vangogh2photo_128/<pasta>.pt` (cerca de 385 MB no total) ([ADR 0003](../adr/0003-dataset-em-memoria.md)).
3. **Normalizar, no batch:** `x / 127.5 - 1`, levando [0, 255] para [-1, 1], a mesma faixa da saída `tanh` do gerador (`to_float`).
4. ***Data augmentation*, só no treino:** espelhamento horizontal com probabilidade 0,5 (`random_batch`). Nunca vertical, porque o céu precisa continuar em cima.

## Como o treino sorteia os dados

- Cada **passo** sorteia 1 foto e 1 pintura, **com reposição** (`torch.randint`).
- Uma **época** é um número de passos igual ao número de fotos usadas (`--photos`, ou 6.287). Como o sorteio é com reposição, uma época não vê cada foto exatamente uma vez. Em média vê cerca de 63% delas.
- As pinturas, com 400 contra milhares de passos, são vistas muitas vezes por época: cerca de 5 vezes cada com `--photos 2000`.

## Limitações para citar no artigo

- **Não há pinturas de teste independentes** (`testA` = `trainA`), então não dá para medir a generalização no domínio das pinturas.
- **Um único artista, 400 obras:** pode não representar todas as fases de Van Gogh.
- **Fotos só de paisagens diurnas do Flickr:** fotos noturnas, de interiores ou com pessoas estão fora da distribuição.
- **Licenças das fotos:** o dataset não informa a licença de cada foto do Flickr.
