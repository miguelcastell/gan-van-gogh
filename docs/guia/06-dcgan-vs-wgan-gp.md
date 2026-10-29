# 6. DCGAN x WGAN-GP: a configuração, linha por linha

## A diferença em uma frase

- **DCGAN:** o discriminador é um **perito** que responde "real ou falsa?". A resposta é uma probabilidade entre 0 e 1.
- **WGAN-GP:** o discriminador vira um **crítico de arte** que dá uma **nota** de quão realista a imagem está. A nota não tem limite.

Por que isso importa: quando o perito da DCGAN fica muito bom, ele responde "falsa, com 100% de certeza" para tudo o que o gerador faz. O gerador não recebe nenhuma pista de *como* melhorar e para de aprender. O crítico da WGAN-GP sempre dá uma nota, mesmo para imagens ruins ("está ruim, mas menos ruim que antes"), então o gerador sempre tem uma direção para seguir.

## A tabela, linha por linha

| | DCGAN | WGAN-GP |
|---|---|---|
| Batch | 128 | 128 |
| Ruído *z* | 100 dimensões | 100 dimensões |
| Otimizador | Adam, lr 2e-4, betas (0.5, 0.999) | Adam, lr 1e-4, betas (0.0, 0.9) |
| Específico | BCE, *label smoothing* opcional | n_critic 5, λ_gp 10, crítico sem BatchNorm |

### Batch: 128 (igual nos dois)

Quantas imagens o modelo vê de uma vez antes de ajustar os pesos. É como corrigir 128 provas e só então decidir o que mudar na aula, em vez de mudar a cada prova.

- Batch maior dá uma estimativa mais estável do erro, mas usa mais memória da GPU.
- Os dois modelos usam o mesmo valor para que a comparação seja justa.

### Ruído z: 100 dimensões (igual nos dois)

O ponto de partida de cada imagem: uma lista de 100 números sorteados. É a "semente" da imagem.

- Cada *z* diferente gera uma imagem diferente. O gerador aprende a transformar esses números em roupas.
- Se você mudar um pouco o *z*, a imagem muda um pouco. Por isso as grades por época usam sempre o mesmo *z* (`fixed_z`): assim dá para ver a *mesma* imagem evoluindo.

### Otimizador: Adam

O otimizador é o algoritmo que ajusta os pesos da rede a cada passo. O Adam tem dois tipos de parâmetro:

- **lr (*learning rate*, taxa de aprendizado):** o tamanho de cada passo. Muito grande, o modelo "pula" a solução e o treino explode. Muito pequeno, o treino demora demais.
- **betas:** o quanto o otimizador lembra dos passos anteriores (o "embalo", ou *momentum*). `beta1` alto significa que ele continua na direção em que vinha. `beta1` baixo significa que ele reage mais ao passo atual.

Por que os valores diferem:

- **DCGAN (lr 2e-4, beta1 0.5):** é a receita do artigo original da DCGAN. O padrão do Adam é `beta1 = 0.9`, mas numa GAN o "alvo" muda o tempo todo (o oponente também aprende), então lembrar demais do passado atrapalha. Por isso o embalo é reduzido para 0.5.
- **WGAN-GP (lr 1e-4, beta1 0.0):** é a receita do artigo da WGAN-GP. Aqui o embalo é zerado, e o passo é menor, porque o crítico é treinado 5 vezes mais (veja `n_critic`).

`2e-4` é notação científica: 2 x 10⁻⁴ = 0,0002.

### Específico da DCGAN

- **BCE (*Binary Cross-Entropy*):** a função de erro para perguntas de sim ou não. Ela pune o discriminador quando ele erra, e pune mais quando ele erra com muita certeza.
- ***Label smoothing* (opcional):** em vez de dizer ao discriminador que as imagens reais são "100% reais" (rótulo 1), dizemos "90% reais" (rótulo 0.9). Isso impede o perito de ficar confiante demais, e o gerador continua recebendo um sinal útil. No código: `--label-smoothing 0.1`.

### Específico da WGAN-GP

- **Distância de Wasserstein:** a ideia por trás da nota do crítico. Imagine as imagens reais como um monte de terra e as imagens geradas como outro. A distância de Wasserstein mede quanto trabalho seria preciso para mover um monte até ficar no formato do outro. Mesmo quando o gerador está muito ruim, essa distância diz "falta tanto". A pergunta "real ou falsa?" só diria "falsa".
- **n_critic = 5:** o crítico treina 5 vezes para cada vez que o gerador treina. A nota só é útil se o crítico for bom, então ele precisa estar sempre "à frente".
- **λ_gp = 10 (*gradient penalty*, penalidade de gradiente):** sem limite, o crítico poderia inflar as notas indefinidamente e o treino perderia o sentido. A penalidade obriga o crítico a ser "suave": imagens parecidas precisam receber notas parecidas. O 10 é o peso dessa regra na função de erro (valor do artigo).
- **Crítico sem BatchNorm:** o BatchNorm normaliza as imagens *em conjunto*, então a nota de uma imagem passaria a depender das outras do mesmo batch. A penalidade de gradiente é calculada *imagem por imagem*, e as duas coisas não combinam. Por isso o crítico não usa BatchNorm (veja `Critic` em `src/models.py`).

## O que esperar ao comparar

| | DCGAN | WGAN-GP |
|---|---|---|
| Velocidade | Mais rápida (1 passo do D por passo do G) | Mais lenta (5 passos do crítico por passo do G, mais o cálculo da penalidade) |
| Curvas de loss | Oscilam e não dizem muito sobre a qualidade | A estimativa de Wasserstein tende a cair conforme as imagens melhoram |
| Risco de colapso de modo | Maior | Menor, em geral |
| Qualidade final | Em bases simples, pode ser parecida | Ganho principal é a estabilidade, não necessariamente a nitidez |

Essas são as expectativas da literatura. O seu experimento vai confirmar ou não, e qualquer resultado é bom material para o relatório.

## Onde está no código

- DCGAN: `src/train_dcgan.py`. Procure `bce`, `real_label` e `opt_d`.
- WGAN-GP: `src/train_wgan_gp.py`. Procure `gradient_penalty`, `n_critic` e `loss_c`.
- Arquiteturas: `src/models.py` (`Discriminator` e `Critic`).

Anterior: [KDD e a exploração dos dados](05-kdd.md)
