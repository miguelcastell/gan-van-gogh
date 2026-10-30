# 6. A CycleGAN por dentro: a configuração, linha por linha

## A configuração

| | Valor |
|---|---|
| Resolução | 128x128 |
| Batch | 1 |
| Gerador | ResNet, 6 blocos residuais, InstanceNorm |
| Discriminador | PatchGAN 70x70 |
| Loss adversarial | LSGAN |
| λ_cycle / λ_identity | 10 / 5 |
| Otimizador | Adam, lr 2e-4, betas (0.5, 0.999), lr caindo linearmente na 2ª metade |
| Image pool | 50 |

Quase tudo vem do artigo da CycleGAN. A única mudança é a resolução, reduzida para caber na GPU e no prazo.

## As três losses

O gerador G (foto → pintura) é treinado para minimizar uma soma de três erros:

```
loss_G = adversarial  +  10 × cycle  +  5 × identity
```

### Adversarial: "pareça um Van Gogh"

O D_pintura dá uma nota para a saída do G. O G é punido quando a nota diz "falsa". Sozinha, essa loss produz pinturas convincentes, mas sem compromisso com a foto de entrada.

### Cycle: "não perca o conteúdo"

A foto vira pintura e a pintura volta a ser foto: `F(G(foto))`. A *cycle loss* é a diferença, pixel a pixel, entre essa foto reconstruída e a original. Também vale no sentido contrário: `G(F(pintura))` deve voltar à pintura.

- O **10** (λ_cycle) é o peso. Ele diz que preservar o conteúdo vale 10 vezes mais que enganar o discriminador.
- Com λ_cycle = 0 (a comparação do projeto), sobra só a adversarial, e nada impede o G de ignorar a foto.

### Identity: "não mexa no que já está certo"

Se o G recebe algo que **já é** uma pintura de Van Gogh, ele deveria devolvê-la quase igual: `G(pintura) ≈ pintura`. Sem essa regra, o G tende a mudar as cores da imagem toda, por exemplo deixando um céu azul alaranjado.

- O **5** (λ_identity) é metade de λ_cycle, como no artigo.

## As peças da arquitetura

### Gerador ResNet com 6 blocos

O gerador tem três partes:

1. **Codificador:** reduz a imagem de 128x128 para 32x32, mas com muito mais canais. É o "resumo" do que tem na foto.
2. **6 blocos residuais:** transformam o resumo. Cada bloco calcula uma **correção** e soma ao que entrou (`saída = entrada + correção`). Isso facilita manter a foto e mudar só o estilo, porque "não mudar nada" é fácil de aprender: basta a correção ser zero.
3. **Decodificador:** volta para 128x128 com 3 canais e termina com `tanh`, que produz valores entre -1 e 1, a mesma faixa das imagens reais.

O artigo usa 9 blocos para 256x256 e 6 para 128x128.

### Discriminador PatchGAN

Em vez de uma nota para a imagem inteira, o PatchGAN devolve uma **grade de notas**. Cada nota julga um pedaço da imagem de cerca de 70x70 pixels: "este pedaço parece pincelada de Van Gogh?".

É exatamente o que interessa aqui: o estilo de um pintor está na **textura local**, nas pinceladas. A composição geral da cena vem da foto, que é preservada pela *cycle loss*.

### InstanceNorm em vez de BatchNorm

Normalizar é ajustar os números de uma camada para ficarem numa faixa estável. O BatchNorm faz isso usando **todas as imagens do batch juntas**. Com batch 1, isso não faz sentido. O InstanceNorm normaliza **cada imagem sozinha**, e ainda remove o contraste e a iluminação específicos daquela foto, o que ajuda a transferir estilo.

## O treino

### Batch 1

Uma imagem por passo. É a receita do artigo, e combina com o InstanceNorm. O treino fica mais "barulhento", mas cada passo é barato.

### LSGAN: erro quadrático em vez de BCE

A GAN original usa a BCE (*Binary Cross-Entropy*), que mede o quanto o perito acerta "real ou falsa". A LSGAN usa o **erro quadrático**: o D tenta dar nota 1 para as reais e 0 para as falsas, e o erro é a distância ao quadrado até esse alvo.

A vantagem: com a BCE, quando o D fica muito confiante, o sinal para o G praticamente some. Com o erro quadrático, uma imagem muito longe do alvo continua gerando um sinal forte, e o treino fica mais estável.

### Adam: lr 2e-4, betas (0.5, 0.999)

- **lr (*learning rate*):** o tamanho de cada passo de ajuste. `2e-4` = 0,0002.
- **beta1 = 0.5:** o quanto o otimizador "lembra" da direção anterior. O padrão do Adam é 0.9, mas numa GAN o alvo muda o tempo todo (o oponente também aprende), então lembrar demais atrapalha.
- **lr caindo na 2ª metade:** nas primeiras épocas o lr é fixo, e depois cai em linha reta até zero. Passos grandes no começo para aprender rápido, e passos cada vez menores no fim para refinar sem bagunçar o que já foi aprendido.

### Image pool de 50

O D_pintura não vê só a saída mais recente do G. Ele vê uma mistura com 50 saídas antigas guardadas. Sem isso, o D só reage à última "moda" do G, e os dois ficam se perseguindo em círculos. Com o histórico, o D tem uma visão mais estável do que o G produz.

## O que esperar ao comparar

| | Completa | Sem *cycle loss* |
|---|---|---|
| Estilo | Pinceladas de Van Gogh | Pode ficar até "mais Van Gogh" |
| Conteúdo da foto | Preservado | Tende a se perder: formas trocadas ou a mesma textura em toda foto |
| Erro de ida e volta | Baixo | Alto |
| FID/KID | — | Pode até ser melhor, porque o FID mede o estilo e não o conteúdo |

A última linha é um bom ponto para o relatório: uma métrica pode melhorar enquanto o resultado piora para o que o usuário quer. Essas são expectativas da literatura, e o seu experimento vai confirmar ou não.

## Onde vai estar no código

- Arquiteturas: `src/models.py` (gerador ResNet e discriminador PatchGAN).
- Treino e losses: `src/train_cyclegan.py`. Procure `lambda_cycle`, `lambda_identity` e `pool`.

Anterior: [KDD e a exploração dos dados](05-kdd.md)
