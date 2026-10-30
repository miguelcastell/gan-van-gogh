# 4. Como aprender (e não só rodar)

O código é escrito com ajuda de IA. Ele roda, mas executar sem entender não ensina nada e dificulta defender o trabalho. Use este ciclo em cada etapa.

## O ciclo

1. **Montar no notebook.** Os notebooks guiados (`notebooks/`) fazem você escrever cada peça. O código completo em `src/` é o gabarito ([ADR 0006](../adr/0006-notebooks-e-scripts.md)).
2. **Responder perguntas sobre ele.** Se não souber responder, volte ao passo 1.
3. **Fazer um experimento pequeno, prevendo o resultado antes de rodar.**
4. **Anotar o que observou** em [`experimentos/`](../experimentos/). Essas notas viram o texto final.

Alternativa: tentar escrever um arquivo sozinho e pedir revisão depois.

## Perguntas para se responder

### models.py
- [ ] Por que a última camada do gerador usa `tanh`?
- [ ] O que um bloco residual faz, e por que ele ajuda a preservar a foto de entrada?
- [ ] Por que a CycleGAN usa InstanceNorm em vez de BatchNorm?
- [ ] O discriminador PatchGAN devolve uma grade de notas em vez de uma nota só. O que cada nota representa?

### train_cyclegan.py
- [ ] Por que existem dois geradores e dois discriminadores, se só queremos foto → pintura?
- [ ] O que a *cycle loss* compara, exatamente? E a *identity loss*?
- [ ] Por que o discriminador às vezes recebe imagens falsas antigas (o *image pool*)?
- [ ] Por que usamos sempre as mesmas fotos de teste para gerar as grades?

### sky.py
- [ ] Por que a máscara tem a borda suavizada, em vez de ser só 0 ou 1?

### evaluate.py
- [ ] O que o FID compara, exatamente? O que ele não consegue captar?
- [ ] Por que o FID fica pouco confiável com só 400 pinturas, e o que o KID faz de diferente?

## Experimentos sugeridos

Ver uma GAN quebrar ensina mais do que vê-la funcionar, e esses testes viram material para a seção "o que deu errado" do relatório.

| Experimento | O que muda | Previsão (preencha antes) |
|---|---|---|
| Sem *cycle loss* | `--sem-cycle` (é a comparação principal) | |
| Sem *identity loss* | `--lambda-identity 0` | |
| *Cycle loss* mais fraca | `--lambda-cycle 2` | |
| Outra seed | `--seed 7` | |

Use `--out` para não sobrescrever o treino principal, ex.: `--out cyclegan_sem_identity`.

## Leitura mínima

- Goodfellow et al. (2014), *Generative Adversarial Nets*: a ideia do jogo.
- Zhu et al. (2017), *CycleGAN*: o ciclo e as losses. Leia as seções 3 e 4 e olhe as figuras.
- Isola et al. (2017), *pix2pix*: de onde vem o discriminador PatchGAN.

Anterior: [Próximos passos](03-proximos-passos.md) · Próximo: [KDD e a exploração dos dados](05-kdd.md)
