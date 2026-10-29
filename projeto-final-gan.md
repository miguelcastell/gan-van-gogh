# Projeto Final 3: GAN — Geração sintética e avaliação de qualidade

**Disciplina:** Aprendizado Não Supervisionado
**Prazo de entrega:** 02/12/2026
**Execução:** 100% local (RTX 3050 6 GB, 16 GB RAM, Intel i5 13ª geração)

---

## 1. O que o enunciado pede

1. Escolher uma base simples para geração (imagens, sinais ou dados tabulares).
2. Treinar uma GAN básica ou uma variação adequada ao problema.
3. Acompanhar a evolução do gerador e do discriminador ao longo do treino.
4. Avaliar diversidade, realismo, estabilidade e exemplos ruins.
5. Discutir riscos de uso, viés, privacidade e limites da geração sintética.

**Entrega:** amostras geradas, curva de treino, comparação e análise ética.

---

## 2. Escopo proposto

| Item | Decisão |
|---|---|
| Dataset | Fashion-MNIST (28x28, escala de cinza, 60 mil imagens de treino) |
| Modelo base | DCGAN |
| Variação para comparação | WGAN-GP (alternativa: DCGAN com spectral norm) |
| Framework | PyTorch |
| Métricas | FID, cobertura de classes por classificador, vizinho mais próximo (memorização) |
| Entregáveis | Notebook/scripts, grades de amostras por época, curvas de loss, tabela comparativa, texto da análise ética |

**Por que Fashion-MNIST:** treina rápido, cabe folgado em 6 GB de VRAM, mostra evolução visível a cada época e tem 10 classes, o que facilita medir diversidade e colapso de modo.

**Extensão opcional (só se sobrar tempo):** repetir o experimento em 64x64 (CelebA ou CIFAR-10 redimensionado) para mostrar o que muda com resolução maior. Com 6 GB de VRAM é possível, mas use batch pequeno (32 a 64) e espere treinos mais longos.

**Fora do escopo:** modelos grandes (StyleGAN, difusão), resoluções acima de 64x64, ajuste fino exaustivo de hiperparâmetros.

---

## 3. O que você precisa levantar antes de começar

### 3.1 Ambiente local
- [ ] Atualizar o driver NVIDIA (verificar com `nvidia-smi`).
- [ ] Python 3.10 ou 3.11 instalado.
- [ ] Ambiente virtual criado (`venv` ou `conda`).
- [ ] PyTorch com suporte a CUDA instalado. O comando exato depende da sua versão de CUDA, gere em https://pytorch.org/get-started/locally/.
- [ ] Testar se a GPU é reconhecida:
  ```python
  import torch
  print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```
- [ ] Instalar as demais dependências:
  ```
  pip install matplotlib numpy pandas scikit-learn tqdm jupyterlab torchmetrics torch-fidelity imageio
  ```
- [ ] Repositório no GitHub criado (versionar código e resultados, sem subir o dataset).

### 3.2 Leitura mínima
- [ ] Goodfellow et al. (2014), *Generative Adversarial Nets*: ideia central do jogo gerador x discriminador.
- [ ] Radford et al. (2015), *DCGAN*: receita da arquitetura.
- [ ] Gulrajani et al. (2017), *Improved Training of Wasserstein GANs*: motivação e mudança na loss.
- [ ] Tutorial oficial de DCGAN do PyTorch (base do código).

### 3.3 Levantamento para a análise ética
Vá guardando as referências desde a primeira semana:
- [ ] 2 ou 3 casos reais de deepfake ou desinformação com imagens sintéticas.
- [ ] Pontos da LGPD relevantes para dados pessoais e dados sintéticos.
- [ ] Estudos sobre viés em modelos generativos e em bases de imagens.
- [ ] Trabalhos sobre memorização e vazamento de dados de treino em modelos generativos.

### 3.4 Confirmar com o professor
- [ ] Formato da entrega (notebook, relatório, apresentação?).
- [ ] Tamanho esperado do texto e critérios de avaliação.
- [ ] Se a entrega é individual ou em grupo.

---

## 4. Configuração recomendada para 6 GB de VRAM

| Parâmetro | Valor sugerido |
|---|---|
| Batch size | 128 (Fashion-MNIST); 32 a 64 se for para 64x64 |
| Dimensão do ruído (z) | 100 |
| Otimizador | Adam, lr = 2e-4, beta1 = 0.5 |
| Normalização das imagens | [-1, 1], saída do gerador com `tanh` |
| Ativações | LeakyReLU (0.2) no discriminador, ReLU no gerador |
| Épocas | 30 a 50 para o DCGAN em Fashion-MNIST |
| WGAN-GP | n_critic = 5, lambda_gp = 10, Adam lr = 1e-4, betas = (0.0, 0.9), sem BatchNorm no crítico |
| Precisão mista (AMP) | Opcional; não é necessária para este tamanho de modelo |

**Dicas para a sua máquina**
- No Windows, use `num_workers=0` no `DataLoader` se tiver erro em notebook; com scripts `.py` e o guard `if __name__ == "__main__":` dá para usar 2 a 4 workers.
- Fashion-MNIST cabe inteiro na RAM, então carregue como tensor e evite gargalo de disco.
- Monitore a VRAM com `nvidia-smi`. Se estourar, reduza o batch size.
- Salve checkpoints a cada poucas épocas (`torch.save`), para não perder um treino se o notebook travar.
- Fixe as seeds (`torch.manual_seed`) para os experimentos serem reproduzíveis.

---

## 5. Estrutura de pastas sugerida

```
projeto-gan/
├── README.md
├── requirements.txt
├── data/                  # dataset baixado (não versionar)
├── notebooks/
│   └── 01_exploracao.ipynb
├── src/
│   ├── models.py          # Gerador, Discriminador/Crítico
│   ├── train_dcgan.py
│   ├── train_wgan_gp.py
│   ├── evaluate.py        # FID, diversidade, vizinho mais próximo
│   └── utils.py
├── outputs/
│   ├── dcgan/
│   │   ├── samples/       # grades por época (ruído fixo)
│   │   ├── checkpoints/
│   │   └── logs.csv       # losses, D(x), D(G(z))
│   └── wgan_gp/
└── report/
    ├── figuras/
    └── analise_etica.md
```

---

## 6. Plano de execução

### Etapa 1: Base e baseline (até 04/11)
- [ ] Ambiente local funcionando com GPU.
- [ ] Baixar o Fashion-MNIST e visualizar amostras reais.
- [ ] Implementar Gerador e Discriminador do DCGAN.
- [ ] Primeiro treino curto (poucas épocas) gerando alguma coisa.

### Etapa 2: DCGAN completo (até 11/11)
- [ ] Treino completo com ruído fixo para gerar uma grade por época.
- [ ] Registrar loss de G e D, além de D(x) e D(G(z)).
- [ ] Montar GIF da evolução das amostras.
- [ ] Salvar checkpoints.

### Etapa 3: Variação e comparação (até 18/11)
- [ ] Implementar WGAN-GP.
- [ ] Treinar com o mesmo orçamento de épocas para comparação justa.
- [ ] Comparar curvas, estabilidade e qualidade visual lado a lado.

### Etapa 4: Avaliação (até 25/11)
- [ ] **Realismo:** FID de cada modelo contra o conjunto de teste.
- [ ] **Diversidade:** treinar um classificador simples no Fashion-MNIST real, classificar as amostras geradas e ver a distribuição entre as 10 classes (detecta colapso de modo).
- [ ] **Memorização:** para cada amostra gerada, achar o vizinho mais próximo no treino (distância L2 ou em espaço de features) e mostrar pares lado a lado.
- [ ] **Exemplos ruins:** separar e explicar as piores amostras (artefatos, formas mistas, classes ausentes).
- [ ] Tabela comparativa DCGAN x WGAN-GP.

### Etapa 5: Análise ética e fechamento (até 02/12)
- [ ] Escrever a análise ética ligando cada ponto ao que foi observado nos experimentos.
- [ ] Escrever a análise crítica (limitações das métricas, o que deu errado, o que faria diferente).
- [ ] README com instruções para reproduzir.
- [ ] Revisão final e entrega (deixar pelo menos 2 dias de folga).

---

## 7. Roteiro da análise ética

Para cada tópico, conecte com evidências do próprio projeto:

1. **Deepfakes e desinformação:** o que mudaria se a mesma técnica fosse aplicada a rostos? Casos reais.
2. **Viés:** o que o gerador aprendeu bem e o que não aprendeu? Há classes sub-representadas nas amostras? A base reflete quais escolhas e exclusões?
3. **Privacidade e memorização:** os resultados do vizinho mais próximo mostram cópias do treino? Implicações para dados pessoais e para a LGPD.
4. **Uso de dados sintéticos como se fossem reais:** riscos de treinar outros modelos com dados gerados, perda de diversidade ao longo de gerações.
5. **Limites das métricas:** o que o FID não captura? Por que boas métricas não garantem qualidade percebida ou ausência de viés.
6. **Medidas de mitigação:** marcação de conteúdo sintético, documentação da base, auditoria de amostras.

---

## 8. Riscos e como contornar

| Risco | Como contornar |
|---|---|
| Discriminador vence cedo e o gerador para de aprender | Reduzir capacidade do D, label smoothing, treinar G mais vezes, ou migrar para WGAN-GP |
| Colapso de modo | Checar distribuição de classes nas amostras, usar WGAN-GP, ajustar lr |
| Treino instável | Fixar seeds, rodar mais de uma vez, comparar curvas |
| VRAM estourando | Reduzir batch size, fechar outros programas que usam a GPU |
| Perder um treino longo | Checkpoints periódicos e logs em CSV |
| FID instável com poucas amostras | Usar pelo menos 10 mil amostras geradas para o cálculo |
| Estourar o prazo | Fechar a Etapa 4 até 25/11 e proteger a última semana para o texto |

---

## 9. Checklist final de entrega

- [ ] Amostras geradas (grades finais de cada modelo, GIF de evolução)
- [ ] Curvas de treino (losses e D(x)/D(G(z)))
- [ ] Comparação DCGAN x WGAN-GP (visual e métricas)
- [ ] Exemplos ruins comentados
- [ ] Análise ética com referências
- [ ] Análise crítica e limitações
- [ ] Código organizado e README para reproduzir
- [ ] Formato de entrega conferido com o professor
