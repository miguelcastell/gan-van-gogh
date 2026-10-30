# Projeto Final 3: GAN — Céu de Van Gogh em fotos reais

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

## 2. A ideia

Tirar uma foto qualquer (por exemplo, de uma estátua) e receber **a mesma foto com o céu pintado no estilo de Van Gogh**, com o resto intacto. A ideia foi sugerida pelo professor.

```
foto → [segmentador pré-treinado] → máscara do céu
foto → [gerador da CycleGAN foto → Van Gogh] → foto toda estilizada
resultado = estilizada onde é céu + original no resto
```

| Item | Decisão |
|---|---|
| Dataset | `vangogh2photo` (pinturas de Van Gogh e fotos de paisagens, sem pares) |
| Modelo | CycleGAN (Zhu et al., 2017), treinada por nós em 128x128 |
| Comparação | CycleGAN completa x CycleGAN **sem a *cycle loss*** |
| Máscara do céu | Segmentador pré-treinado (não treinado por nós) |
| Framework | PyTorch |
| Métricas | FID/KID contra pinturas reais, preservação do conteúdo, vizinho mais próximo (memorização) |
| Entregáveis | Código, notebook, grades por época, curvas de loss, comparação, demonstração com fotos próprias, análise ética |

**Por que CycleGAN:** ela aprende a converter imagens de um estilo para outro **sem precisar de pares**. Não existe uma foto da cena exata de cada pintura de Van Gogh, então esse é o caso típico dela. A *cycle loss* (foto → pintura → foto deve voltar à original) é o que obriga a estátua a continuar sendo a mesma estátua.

**Limitação importante:** a CycleGAN aprende o **estilo geral** de Van Gogh a partir de centenas de pinturas, não a *Noite Estrelada* em si. O céu terá as pinceladas e as cores dele, mas não os redemoinhos exatos daquele quadro.

**Fora do escopo:** treinar o segmentador, resoluções acima de 256x256, modelos de difusão, ajuste fino exaustivo de hiperparâmetros.

---

## 3. O que você precisa levantar antes de começar

### 3.1 Ambiente local
- [x] Driver NVIDIA atualizado (`nvidia-smi` mostra a RTX 3050, driver 610).
- [x] Ambiente virtual `.venv` criado.
- [x] **PyTorch com CUDA no venv:** `2.15.0+cu132`. O driver suporta CUDA 13.3.
- [x] Testar se a GPU é reconhecida:
  ```python
  import torch
  print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```
- [x] Instalar as demais dependências: `pip install -r requirements.txt`.

### 3.2 Leitura mínima
- [ ] Goodfellow et al. (2014), *Generative Adversarial Nets*: a ideia do jogo gerador x discriminador.
- [ ] Zhu et al. (2017), *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (CycleGAN): a *cycle loss* e a *identity loss*.
- [ ] Isola et al. (2017), *pix2pix*: de onde vem o discriminador PatchGAN.
- [ ] Mao et al. (2017), *Least Squares GAN*: a loss adversarial usada pela CycleGAN.

### 3.3 Levantamento para a análise ética
- [ ] Casos de imitação de estilo de artistas vivos por IA generativa e a reação deles.
- [ ] Casos reais de fotos manipuladas usadas como desinformação.
- [ ] Direitos autorais: domínio público (Van Gogh) x artistas vivos; pontos da LGPD sobre fotos de pessoas.
- [ ] Trabalhos sobre memorização em modelos generativos.

### 3.4 Confirmar com o professor
- [ ] Formato da entrega (notebook, relatório, apresentação?).
- [ ] Tamanho esperado do texto e critérios de avaliação.
- [ ] Se a entrega é individual ou em grupo.

---

## 4. Configuração recomendada para 6 GB de VRAM

| Parâmetro | Valor | Origem |
|---|---|---|
| Resolução | 128x128 (o artigo usa 256x256) | Reduzido para caber no prazo |
| Batch size | 1 | Artigo |
| Gerador | ResNet com 6 blocos residuais, InstanceNorm | Artigo (versão para 128x128) |
| Discriminador | PatchGAN 70x70, InstanceNorm | Artigo |
| Loss adversarial | LSGAN (erro quadrático) | Artigo |
| λ_cycle | 10 | Artigo |
| λ_identity | 5 (metade de λ_cycle) | Artigo |
| Otimizador | Adam, lr = 2e-4, betas = (0.5, 0.999) | Artigo |
| Agenda do lr | Constante na 1ª metade das épocas, cai linearmente até 0 na 2ª | Artigo |
| Image pool | 50 imagens falsas antigas para o discriminador | Artigo |
| Épocas | A definir. Medido: ~130 ms por passo, ou ~13 min por época com todas as fotos | — |

**Dicas para a sua máquina**
- As imagens cabem na RAM (as 7.838 imagens em 128x128 coloridas ocupam uns 385 MB em `uint8`), então carregue tudo de uma vez e evite gargalo de disco.
- Monitore a VRAM com `nvidia-smi`.
- Salve checkpoints a cada poucas épocas (`torch.save`): o treino vai levar horas.
- Fixe as seeds (`torch.manual_seed`) para os experimentos serem reproduzíveis.

---

## 5. Estrutura de pastas

```
gan-van-gogh/
├── README.md
├── requirements.txt
├── data/vangogh2photo/    # dataset baixado (não versionar)
│   ├── trainA/ testA/     # pinturas de Van Gogh
│   └── trainB/ testB/     # fotos
├── notebooks/
│   ├── 01_exploracao.ipynb   # dados (KDD 1 a 3)
│   └── 02_cyclegan.ipynb     # monta as peças e faz um treino curto
├── src/
│   ├── download_data.py   # baixa o dataset
│   ├── models.py          # Gerador ResNet, Discriminador PatchGAN
│   ├── train_cyclegan.py  # treino, com opção de desligar a cycle loss
│   ├── sky.py             # máscara do céu e composição
│   ├── stylize.py         # aplica numa foto própria
│   ├── evaluate.py        # FID/KID, preservação do conteúdo, vizinho mais próximo
│   └── utils.py
├── outputs/
│   ├── cyclegan/
│   │   ├── samples/       # grades por época (fotos fixas)
│   │   ├── checkpoints/
│   │   └── logs.csv       # losses de G e D
│   └── cyclegan_sem_cycle/
└── report/
    ├── figuras/
    └── analise_etica.md
```

---

## 6. Plano de execução

### Etapa 1: Ambiente e dados (até 04/11)
- [ ] PyTorch com CUDA funcionando.
- [x] Baixar o `vangogh2photo`.
- [ ] Explorar os dados no notebook 01.
- [ ] Montar o gerador, o discriminador e as losses no notebook 02 (o gabarito está em `src/`).
- [ ] Primeiro treino curto, para medir o tempo de uma época.

### Etapa 2: CycleGAN completa (até 11/11)
- [ ] Treino completo com as mesmas fotos de teste gerando uma grade por época.
- [ ] Registrar as losses (adversarial, cycle e identity) de cada gerador e discriminador.
- [ ] Montar GIF da evolução.
- [ ] Salvar checkpoints.

### Etapa 3: Comparação e máscara do céu (até 18/11)
- [ ] Treinar a versão sem *cycle loss* com o mesmo orçamento de épocas.
- [ ] Comparar curvas e resultados lado a lado.
- [ ] Integrar o segmentador de céu e a composição com borda suavizada.
- [ ] Testar com fotos próprias (a estátua).

### Etapa 4: Avaliação (até 25/11)
- [ ] **Realismo do estilo:** FID e KID entre as fotos convertidas e as pinturas reais.
- [ ] **Preservação do conteúdo:** quanto a foto muda ao ir e voltar (foto → pintura → foto).
- [ ] **Memorização:** para cada saída, achar a pintura mais parecida e mostrar os pares. Referência: a distância entre pinturas reais diferentes (não há pinturas de teste separadas: `testA` = `trainA`).
- [ ] **Exemplos ruins:** máscaras erradas, artefatos, céus que viraram outra coisa.
- [ ] Tabela comparativa: completa x sem *cycle loss*.

### Etapa 5: Análise ética e fechamento (até 02/12)
- [ ] Escrever a análise ética ligando cada ponto ao que foi observado nos experimentos.
- [ ] Escrever a análise crítica (limitações das métricas, o que deu errado, o que faria diferente).
- [ ] README com instruções para reproduzir.
- [ ] Revisão final e entrega (deixar pelo menos 2 dias de folga).

---

## 7. Roteiro da análise ética

Para cada tópico, conecte com evidências do próprio projeto:

1. **Imitação de estilo e autoria:** Van Gogh é domínio público, mas a mesma técnica copia o estilo de artistas vivos sem permissão. Quem é o autor da imagem gerada?
2. **Manipulação de fotos e desinformação:** trocar o céu de uma foto é editar a realidade. O que muda quando a técnica é aplicada a fotos jornalísticas ou a rostos?
3. **Viés:** as fotos do dataset são paisagens, e o estilo vem de um único pintor. Que tipos de foto o modelo trata mal? Que parte da obra de Van Gogh ele representa?
4. **Memorização:** o vizinho mais próximo mostra pedaços copiados das pinturas?
5. **Limites das métricas:** com poucas centenas de pinturas, o FID é pouco confiável. O que nenhuma métrica captura sobre "parecer Van Gogh"?
6. **Medidas de mitigação:** marcação de conteúdo sintético, documentação da base, consentimento de artistas.

---

## 8. Riscos e como contornar

| Risco | Como contornar |
|---|---|
| Treino longo demais para o prazo | Medir o tempo de uma época primeiro; usar um subconjunto das fotos e menos épocas |
| Desequilíbrio entre domínios (~400 pinturas x ~6 mil fotos) | Sortear um subconjunto de fotos por época; observar se o discriminador de pinturas decora o treino |
| Gerador muda o conteúdo da foto | É o que a *cycle loss* evita; a versão sem ela mostra o problema |
| Cores mudando em áreas que não deveriam | *Identity loss* |
| Segmentador erra o céu | Separar esses casos como exemplos ruins; suavizar a borda da máscara |
| Perder um treino longo | Checkpoints periódicos e logs em CSV |
| FID instável com poucas pinturas | Reportar também o KID, que funciona melhor com poucas amostras |
| Estourar o prazo | Fechar a Etapa 4 até 25/11 e proteger a última semana para o texto |

---

## 9. Checklist final de entrega

- [ ] Resultados (fotos com céu de Van Gogh, grades finais, GIF de evolução)
- [ ] Curvas de treino (losses dos geradores e discriminadores)
- [ ] Comparação: completa x sem *cycle loss* (visual e métricas)
- [ ] Exemplos ruins comentados
- [ ] Análise ética com referências
- [ ] Análise crítica e limitações
- [ ] Código organizado e README para reproduzir
- [ ] Formato de entrega conferido com o professor
