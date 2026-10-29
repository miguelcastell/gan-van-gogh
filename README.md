# GAN — Geração sintética de imagens e avaliação de qualidade

Treino e comparação de **Redes Adversárias Generativas (GANs)** para gerar imagens sintéticas de roupas a partir do dataset [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), com avaliação de realismo, diversidade, estabilidade e memorização, e uma discussão sobre os riscos do uso de dados sintéticos.

> 🚧 Em desenvolvimento.

## O que é uma GAN

Uma GAN é formada por duas redes neurais que treinam uma contra a outra:

- **Gerador (G):** recebe um vetor de números aleatórios (o ruído *z*) e tenta transformá-lo numa imagem que pareça real. Funciona como um falsificador.
- **Discriminador (D):** recebe uma imagem e tenta dizer se ela é real (do dataset) ou falsa (criada pelo gerador). Funciona como um perito.

A cada passo, o discriminador aprende a separar imagens reais de falsas, e o gerador usa o retorno do discriminador para enganá-lo melhor. Quando o treino dá certo, o gerador passa a produzir imagens novas, que não estão no dataset, mas seguem o mesmo padrão.

O treino é instável por natureza: se o discriminador ficar bom rápido demais, o gerador para de aprender. O gerador também pode encontrar um "atalho" e produzir sempre o mesmo tipo de imagem (**colapso de modo**). Observar e medir esses problemas é parte central do projeto.

## O que o projeto faz

1. **Exploração dos dados** seguindo o processo KDD: qualidade, balanceamento, padrões por classe e projeção 2D.
2. **DCGAN:** a arquitetura clássica de GAN com camadas convolucionais.
3. **WGAN-GP:** uma variação que troca o discriminador por um "crítico" e usa a distância de Wasserstein com *gradient penalty*, buscando um treino mais estável.
4. **Acompanhamento do treino:** curvas de loss, saídas do discriminador e uma grade de imagens por época, gerada sempre a partir do mesmo ruído para mostrar a evolução.
5. **Avaliação e comparação dos dois modelos:**
   - **Realismo:** FID (*Fréchet Inception Distance*) contra imagens reais.
   - **Diversidade:** um classificador mede quantas imagens geradas caem em cada uma das 10 classes, o que detecta colapso de modo.
   - **Memorização:** busca da imagem de treino mais parecida com cada imagem gerada, para verificar se o modelo copia em vez de gerar.
   - **Exemplos ruins:** as amostras com mais artefatos ou formas misturadas, com a explicação do porquê.
6. **Análise ética:** deepfakes, viés, privacidade e os limites das métricas e da geração sintética.

## Estrutura

```
src/
  models.py            Gerador, Discriminador, Crítico e classificador auxiliar
  utils.py             seeds, carga dos dados, logs, grades de imagens, checkpoints, GIF
  train_dcgan.py       treino da DCGAN
  train_wgan_gp.py     treino da WGAN-GP
  train_classifier.py  classificador usado para medir diversidade
  evaluate.py          FID, distribuição de classes, memorização, piores amostras
  plots.py             curvas de treino e tabela comparativa
notebooks/             exploração dos dados
outputs/<modelo>/      grades por época, logs.csv, metrics.json, evolucao.gif
report/                figuras, tabela comparativa e análise ética
docs/                  guia de estudo, decisões (ADRs), boas práticas e diário de experimentos
```

## Como rodar

Requisitos: Python 3.10+ e, de preferência, uma GPU NVIDIA com CUDA. O projeto foi pensado para caber em 6 GB de VRAM.

```powershell
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate

# PyTorch com CUDA (gere o comando certo para a sua máquina em https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Todos os comandos rodam a partir da raiz do projeto. O dataset é baixado automaticamente em `data/` na primeira execução.

```powershell
python -m src.train_dcgan --epochs 40
python -m src.train_wgan_gp --epochs 40
python -m src.train_classifier
python -m src.evaluate --model dcgan
python -m src.evaluate --model wgan_gp
python -m src.plots
```

## Configuração

| | DCGAN | WGAN-GP |
|---|---|---|
| Batch | 128 | 128 |
| Ruído *z* | 100 dimensões | 100 dimensões |
| Otimizador | Adam, lr 2e-4, betas (0.5, 0.999) | Adam, lr 1e-4, betas (0.0, 0.9) |
| Específico | BCE, *label smoothing* opcional | n_critic 5, λ_gp 10, crítico sem BatchNorm |

As seeds são fixas (`--seed`) para os experimentos serem reproduzíveis.

## Documentação

[`docs/README.md`](docs/README.md) reúne o guia de estudo, as decisões técnicas (ADRs), as boas práticas e o diário de experimentos.

## Referências

- Goodfellow et al. (2014). *Generative Adversarial Nets*.
- Radford, Metz e Chintala (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (DCGAN).
- Gulrajani et al. (2017). *Improved Training of Wasserstein GANs* (WGAN-GP).
- Xiao, Rasul e Vollgraf (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*.
