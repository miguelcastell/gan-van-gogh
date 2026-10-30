# GAN — Céu de Van Gogh em fotos reais

Uma **CycleGAN** aprende a converter fotos no estilo de pintura de **Van Gogh**. Ela é combinada com um segmentador de céu para pegar uma foto qualquer, por exemplo de uma estátua, e devolver a mesma foto com **o céu pintado como Van Gogh pintaria** e o resto intacto. O projeto inclui a avaliação do estilo, da preservação do conteúdo e da memorização, e uma discussão sobre os riscos da imitação de estilo e da manipulação de imagens.

> 🚧 Em desenvolvimento.

## Como funciona

```
foto → [segmentador pré-treinado] → máscara do céu
foto → [gerador da CycleGAN foto → Van Gogh] → foto toda estilizada
resultado = estilizada onde é céu + original no resto
```

### O que é uma GAN

Uma GAN é formada por duas redes neurais que treinam uma contra a outra:

- **Gerador (G):** cria imagens tentando enganar o discriminador. Funciona como um falsificador.
- **Discriminador (D):** recebe uma imagem e tenta dizer se ela é real ou criada pelo gerador. Funciona como um perito.

### O que a CycleGAN acrescenta

Aqui o gerador não parte de ruído: ele recebe uma **foto** e devolve uma **pintura**. A CycleGAN treina dois pares ao mesmo tempo:

- **G:** foto → pintura, julgado por **D_pintura** ("isso parece um Van Gogh de verdade?");
- **F:** pintura → foto, julgado por **D_foto** ("isso parece uma foto de verdade?").

Sozinho, o G poderia enganar o D_pintura devolvendo qualquer pintura bonita, sem relação com a foto. A ***cycle loss*** impede isso: a foto convertida em pintura e de volta em foto, `F(G(foto))`, tem que ficar igual à original. Assim o conteúdo (a estátua, o prédio, o horizonte) é preservado e só o estilo muda.

O treino não precisa de pares foto-pintura da mesma cena, que não existem.

## O que o projeto faz

1. **Exploração dos dados** seguindo o processo KDD: tamanhos, cores, desequilíbrio entre pinturas e fotos.
2. **CycleGAN** treinada no dataset `vangogh2photo` em 128x128.
3. **Comparação:** CycleGAN completa x CycleGAN sem a *cycle loss*, para mostrar o papel dela.
4. **Acompanhamento do treino:** curvas das losses e uma grade por época, sempre com as mesmas fotos de teste.
5. **Máscara do céu:** um segmentador pré-treinado separa o céu, e a composição aplica o estilo só nele.
6. **Avaliação:**
   - **Estilo:** FID e KID entre as fotos convertidas e as pinturas reais.
   - **Preservação do conteúdo:** diferença entre a foto original e a foto que foi e voltou (foto → pintura → foto).
   - **Memorização:** a pintura de treino mais parecida com cada saída, para ver se o modelo copia pedaços.
   - **Exemplos ruins:** máscaras erradas, artefatos e céus que perderam o sentido.
7. **Análise ética:** imitação de estilo e autoria, manipulação de fotos, viés e os limites das métricas.

## Estrutura

```
src/
  models.py            Gerador ResNet e Discriminador PatchGAN
  utils.py             seeds, carga dos dados, logs, grades de imagens, checkpoints, GIF
  download_data.py     baixa o dataset vangogh2photo
  train_cyclegan.py    treino da CycleGAN (com opção de desligar a cycle loss)
  sky.py               máscara do céu e composição
  stylize.py           aplica o céu de Van Gogh numa foto própria
  evaluate.py          FID/KID, preservação do conteúdo, memorização, piores saídas
  plots.py             curvas de treino e tabela comparativa
notebooks/             guiados: 01 exploração dos dados, 02 montar e treinar a CycleGAN
outputs/<experimento>/ grades por época, logs.csv, metrics.json, evolucao.gif
report/                figuras, tabela comparativa e análise ética
docs/                  guia de estudo, decisões (ADRs), boas práticas e diário de experimentos
```

## Como rodar

Requisitos: Python 3.10+ e uma GPU NVIDIA com CUDA. O projeto foi pensado para caber em 6 GB de VRAM.

### 1. Instalar (uma vez)

No PowerShell, na pasta do projeto:

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate

# PyTorch com CUDA (gere o comando certo para a sua máquina em https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
nbstripout --install          # o git passa a ignorar as saídas dos notebooks

python -m src.download_data     # baixa o dataset (~300 MB) em data/vangogh2photo/
```

O dataset também pode ser baixado à mão em http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/vangogh2photo.zip, extraindo dentro de `data/`.

**Toda vez que abrir um terminal novo**, ative o ambiente virtual antes de qualquer comando: `.venv\Scripts\activate`. Vai aparecer `(.venv)` no começo da linha. Todos os comandos rodam a partir da raiz do projeto.

### 2. Notebook ou terminal?

O projeto usa os dois, para coisas diferentes ([ADR 0006](docs/adr/0006-notebooks-e-scripts.md)):

| | Notebooks (`notebooks/`) | Scripts no terminal (`src/`) |
|---|---|---|
| Para que serve | Aprender: explorar os dados e montar cada peça da CycleGAN, com exercícios guiados | Treinar o modelo de verdade e avaliar |
| Quanto treina | ~300 passos (cerca de 1 minuto) | Dezenas de milhares de passos (algumas horas) |
| Se o PC dormir ou a janela fechar | Perde o progresso | Continua do último checkpoint com `--resume` |
| Salva o modelo? | Não | Sim, em `outputs/<experimento>/checkpoints/` |

O código dos scripts é o mesmo que você monta nos notebooks, já completo, e serve de gabarito. A ordem recomendada é: notebook 01, notebook 02 e depois o terminal.

### 3. Notebooks

```powershell
jupyter lab
```

Abra, em ordem:
1. `notebooks/01_exploracao.ipynb`: os dados (KDD, fases 1 a 3).
2. `notebooks/02_cyclegan.ipynb`: monta o gerador, o discriminador e as losses e faz um treino curto, com e sem *cycle loss*.

### 4. Treino no terminal

```powershell
# Teste rápido (~1 min): confere se roda e mostra o tempo por passo
python -m src.train_cyclegan --epochs 1 --photos 300

# Treino completo (~3 h) e a comparação sem cycle loss (~3 h)
python -m src.train_cyclegan --epochs 40 --photos 2000
python -m src.train_cyclegan --epochs 40 --photos 2000 --sem-cycle

# Se o treino for interrompido, continue do último checkpoint salvo
python -m src.train_cyclegan --epochs 40 --photos 2000 --resume outputs/cyclegan/checkpoints/epoch_015.pt
```

| Opção | O que faz |
|---|---|
| `--epochs` | Quantas épocas (voltas pelos dados) |
| `--photos` | Quantas fotos de treino usar. Uma época = esse número de passos |
| `--sem-cycle` | Desliga a *cycle loss* e salva em `outputs/cyclegan_sem_cycle/` |
| `--resume` | Continua um treino a partir de um checkpoint |
| `--amp` | Precisão mista (bf16): um pouco mais rápido e usa menos memória da GPU |
| `--seed` | Semente aleatória, para o experimento ser reproduzível |
| `--out` | Nome da pasta em `outputs/`, para não sobrescrever outro treino |

As outras opções (λ, lr...) aparecem com `python -m src.train_cyclegan --help`.

**Enquanto treina**, o terminal mostra uma linha por época com as losses e o tempo. Em `outputs/cyclegan/`:
- `samples/epoch_NNN.png`: uma grade por época. Linha 1 = fotos de teste, linha 2 = convertidas em pintura, linha 3 = a volta para foto;
- `logs.csv`: as losses ao longo do treino;
- `checkpoints/`: o modelo salvo a cada 5 épocas;
- `evolucao.gif`: a animação das grades, criada no fim.

### 5. Avaliar e usar

```powershell
python -m src.evaluate --model cyclegan             # métricas (metrics.json) e figuras em report/figuras/
python -m src.evaluate --model cyclegan_sem_cycle
python -m src.plots                                 # gráficos das losses e tabela comparativa
python -m src.stylize minha_foto.jpg                # céu de Van Gogh numa foto sua, em outputs/stylize/
```

Na primeira vez, o `stylize` baixa o segmentador de céu (SegFormer-B0, cerca de 15 MB) do Hugging Face. Use `--sem-mascara` para estilizar a foto inteira.

## Configuração

| | Valor |
|---|---|
| Resolução | 128x128 |
| Batch | 1 |
| Gerador | ResNet, 6 blocos residuais, InstanceNorm |
| Discriminador | PatchGAN 70x70 |
| Loss adversarial | LSGAN |
| λ_cycle / λ_identity | 10 / 5 |
| Otimizador | Adam, lr 2e-4, betas (0.5, 0.999), lr caindo linearmente na 2ª metade |
| Segmentador de céu | SegFormer-B0 treinado no ADE20K (pré-treinado, não é treinado aqui) |

As seeds são fixas (`--seed`) para os experimentos serem reproduzíveis.

## Documentação

[`docs/README.md`](docs/README.md) reúne o guia de estudo, as decisões técnicas (ADRs), as boas práticas e o diário de experimentos.

## Referências

- Goodfellow et al. (2014). *Generative Adversarial Nets*.
- Zhu, Park, Isola e Efros (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (CycleGAN).
- Isola et al. (2017). *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix, PatchGAN).
- Mao et al. (2017). *Least Squares Generative Adversarial Networks*.
