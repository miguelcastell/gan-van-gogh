# 4. Como aprender (e não só rodar)

O código inicial foi gerado com ajuda de IA. Ele roda, mas executar sem entender não ensina nada e dificulta defender o trabalho. Use este ciclo em cada etapa.

## O ciclo

1. **Ler com explicação.** Pegar um arquivo e entender trecho por trecho. Comece pelo `src/models.py`: é o mais curto e o centro de tudo.
2. **Responder perguntas sobre ele.** Se não souber responder, volte ao passo 1.
3. **Fazer um experimento pequeno, prevendo o resultado antes de rodar.**
4. **Anotar o que observou** em [`experimentos/`](../experimentos/). Essas notas viram o texto final.

Alternativa: tentar escrever um arquivo sozinho (o `models.py`, por exemplo) e pedir revisão depois.

## Perguntas para se responder

### models.py
- [ ] Por que a última camada do gerador usa `tanh`?
- [ ] Por que o discriminador usa `LeakyReLU` e o gerador usa `ReLU`?
- [ ] Como um vetor de 100 números vira uma imagem 28x28? Siga as dimensões camada por camada.
- [ ] Por que o crítico da WGAN-GP não pode ter BatchNorm?

### train_dcgan.py
- [ ] Por que o ruído `fixed_z` é sempre o mesmo para gerar as grades?
- [ ] Por que usamos `fake.detach()` ao treinar o D, mas não ao treinar o G?
- [ ] O que significa D(x) = 0,9 e D(G(z)) = 0,1? E os dois perto de 0,5?

### evaluate.py
- [ ] O que o FID compara, exatamente? O que ele não consegue captar?
- [ ] Por que comparamos a distância "gerado → treino" com "teste → treino"?

## Experimentos sugeridos

Ver uma GAN quebrar ensina mais do que vê-la funcionar, e esses testes viram material para a seção "o que deu errado" do relatório.

| Experimento | Como rodar | Previsão (preencha antes) |
|---|---|---|
| Learning rate 10x maior | `--lr 2e-3` | |
| Label smoothing | `--label-smoothing 0.1` | |
| Discriminador menor | alterar `ndf` em `models.py` | |
| Outra seed | `--seed 7` | |

Use `--out` para não sobrescrever o treino principal, ex.: `--out dcgan_lr_alto`.

## Leitura mínima

- Goodfellow et al. (2014), *Generative Adversarial Nets*: a ideia do jogo.
- Radford et al. (2015), *DCGAN*: a receita da arquitetura.
- Gulrajani et al. (2017), *Improved Training of Wasserstein GANs*: por que mudar a loss.
- Tutorial oficial de DCGAN do PyTorch.

Anterior: [Próximos passos](03-proximos-passos.md) · Próximo: [KDD e a exploração dos dados](05-kdd.md)
