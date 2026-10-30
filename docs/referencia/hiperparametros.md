# Ficha de hiperparâmetros e configuração

Todos os valores usados pelo código, com a origem de cada um. Conferidos no código em 2026-10-29 (commit `fed4c39`). Cada treino salva os argumentos que usou dentro do checkpoint (`args`), então o valor exato de um experimento sempre pode ser recuperado de lá.

## Treino (`src/train_cyclegan.py`)

| Parâmetro | Opção | Padrão | Origem / motivo |
|---|---|---|---|
| Épocas | `--epochs` | 40 | Escolha nossa, pelo prazo. O artigo usa 200 |
| Início da queda do lr | `--decay-start` | metade das épocas (20) | Artigo: lr constante na 1ª metade, queda linear na 2ª. Com 40 épocas: lr = 2e-4 nas épocas 1 a 21, depois cai 5% do valor inicial por época até 1e-5 na época 40 (`lr_factor` em `src/train_cyclegan.py`; conferido no `logs.csv` do [primeiro treino](../experimentos/2026-10-29-primeiro-treino.md)) |
| Fotos por época | `--photos` | todas (6.287) | Escolha nossa. Subconjunto sorteado com a seed |
| Resolução | `--size` | 128 | Escolha nossa, pelo tempo. O artigo usa 256 |
| Batch | `--batch-size` | 1 | Artigo |
| Learning rate | `--lr` | 2e-4 | Artigo |
| Adam β1 / β2 | `--beta1` | 0,5 / 0,999 | Artigo |
| λ_cycle | `--lambda-cycle` | 10 | Artigo |
| λ_identity | `--lambda-identity` | 5 (= 0,5 × λ_cycle) | Artigo |
| Sem *cycle loss* | `--sem-cycle` | desligado | A comparação do projeto: λ_cycle = 0 e a *identity* continua ([ADR 0002](../adr/0002-dataset-e-modelos.md)) |
| Image pool | `--pool-size` | 50 | Artigo |
| Blocos residuais | `--n-blocks` | 6 | Artigo (versão para 128x128) |
| Precisão mista | `--amp` | desligado | bf16, opcional ([experimento de tempo](../experimentos/2026-10-29-tempo-por-passo.md)) |
| Checkpoint a cada | `--ckpt-every` | 5 épocas | — |
| Log a cada | `--log-every` | 100 passos | — |
| Seed | `--seed` | 42 | Fixa o subconjunto de fotos, os pesos iniciais e os sorteios |
| Pasta de saída | `--out` | `cyclegan` ou `cyclegan_sem_cycle` | — |
| `cudnn.benchmark` | — | sempre ligado | Mais que dobra a velocidade |

Otimizadores: um Adam para G e F juntos e outro para D_pintura e D_foto juntos. Os dois seguem a mesma agenda de lr.

## Losses

```
loss_G = adv(G) + adv(F) + λ_cycle × [L1(F(G(foto)), foto) + L1(G(F(pintura)), pintura)]
                         + λ_identity × [L1(G(pintura), pintura) + L1(F(foto), foto)]

loss_D = 0,5 × [MSE(D(real), 1) + MSE(D(falsa do pool), 0)]     (para cada D)
adv    = MSE(D(falsa), 1)                                       (LSGAN)
```

## Arquitetura (`src/models.py`)

| | Gerador (ResNet) | Discriminador (PatchGAN) |
|---|---|---|
| Parâmetros | 7.837.699 | 2.764.737 |
| Entrada → saída | (3, 128, 128) → (3, 128, 128) | (3, 128, 128) → (1, 14, 14) |
| Filtros base | `ngf` = 64 | `ndf` = 64 |
| Estrutura | conv 7x7 → 2 convs com stride 2 (até 32x32, 256 canais) → 6 blocos residuais → 2 convs transpostas → conv 7x7 → `tanh` | 3 convs com stride 2 → 1 conv com stride 1 → conv de saída. Cada nota cobre ~70x70 pixels |
| Normalização | InstanceNorm | InstanceNorm (exceto na 1ª camada) |
| Ativação | ReLU | LeakyReLU (0,2) |
| Borda | ReflectionPad | padding com zeros |
| Inicialização | Normal(0; 0,02) nos pesos das convoluções, viés 0 | igual |

No total, a CycleGAN tem 2 geradores e 2 discriminadores: 21.204.872 parâmetros.

## Avaliação (`src/evaluate.py`)

| Parâmetro | Valor |
|---|---|
| Seed | 123 (`--seed`) |
| Checkpoint | o último salvo (`--checkpoint` para escolher outro) |
| Features para FID/KID | Inception v3, 2048 dimensões (torchmetrics) |
| KID | `subset_size` = 100 |
| Distância de memorização | L2 nos pixels em [-1, 1], 128x128, com a conta exata |

## Máscara do céu e demonstração (`src/sky.py`, `src/stylize.py`)

| Parâmetro | Valor |
|---|---|
| Segmentador | `nvidia/segformer-b0-finetuned-ade-512-512` (SegFormer-B0, ADE20K, 150 classes, classe `sky`). Pré-treinado, não é treinado aqui |
| Máscara | Probabilidade da classe céu (softmax), não 0/1 |
| Suavização da borda | Desfoque gaussiano com kernel de 1% do maior lado da foto (`blur` = 0,01) |
| Lado menor na entrada do gerador | `--size` = 256 |
| Foto máxima | `--max-side` = 1024 (fotos maiores são reduzidas) |
| Composição | `máscara × estilizada + (1 − máscara) × original` |

## Ambiente

| | |
|---|---|
| GPU | NVIDIA RTX 3050 6 GB Laptop, driver 610.60 (CUDA até 13.3) |
| Python | 3.12.10 |
| PyTorch | 2.15.0+cu132 |
| torchvision | 0.29.2+cu132 |
| transformers | 5.22.0 |
| torchmetrics | 1.9.0 (FID e KID) |
| torch-fidelity | 0.4.0 |
| numpy / pandas / scikit-learn / matplotlib | 2.5.4 / 3.1.0 / 1.9.1 / 3.11.2 |
| Lista completa | `requirements.txt` (sem versões fixas; as versões acima são as usadas) |
