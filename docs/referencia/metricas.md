# Ficha das métricas

O que cada número registrado pelo projeto significa, como é calculado e como ler. Use esta ficha para descrever a metodologia de avaliação no artigo.

## Durante o treino: `outputs/<experimento>/logs.csv`

Uma linha a cada `--log-every` passos (padrão 100). Os valores são do **último passo** antes do registro, não uma média. Como o batch é 1, eles oscilam muito: o `src/plots.py` desenha a média móvel de 20 registros.

| Coluna | O que é | Como ler |
|---|---|---|
| `epoch`, `step` | Época e passo global | — |
| `lr` | *Learning rate* da época | Constante na 1ª metade, depois cai em linha reta |
| `loss_g_adv` | Loss adversarial de G (foto → pintura): `MSE(D_pintura(G(foto)), 1)` | Quanto o D_pintura ainda percebe que a pintura é falsa |
| `loss_f_adv` | Loss adversarial de F (pintura → foto) | Igual, para o D_foto |
| `loss_cycle` | `L1(F(G(foto)), foto) + L1(G(F(pintura)), pintura)` | Quanto se perde na ida e volta. Deve cair. Sem peso λ |
| `loss_identity` | `L1(G(pintura), pintura) + L1(F(foto), foto)` | Quanto o gerador mexe no que já está no domínio de chegada. Sem peso λ |
| `loss_d_pintura`, `loss_d_foto` | `0,5 × [MSE(D(real), 1) + MSE(D(falsa), 0)]` | **0,25** = D chutando 0,5 para tudo. Perto de **0** = D venceu, e o gerador para de receber sinal útil |

## Depois do treino: `outputs/<experimento>/metrics.json`

Calculado pelo `src/evaluate.py` sobre as **751 fotos de teste** (`testB`), que o modelo nunca viu. Referência de pinturas: as **400 pinturas** (`trainA`), porque não existem pinturas de teste separadas ([ficha dos dados](dados.md)).

| Campo | O que é | Melhor quando |
|---|---|---|
| `model`, `checkpoint`, `lambda_cycle` | Qual modelo, qual checkpoint e com que λ_cycle foi treinado | — |
| `n_fotos`, `n_pinturas` | Quantas imagens entraram (751 e 400) | — |
| `fid` | FID entre G(fotos de teste) e as pinturas | Menor |
| `kid`, `kid_std` | KID entre G(fotos de teste) e as pinturas: média e desvio padrão entre subconjuntos | Menor |
| `fid_fotos_originais`, `kid_fotos_originais` | Os mesmos, com as fotos **sem conversão** | É a referência: o modelo deve ficar bem abaixo disso |
| `cycle_l1` | Média de `|F(G(foto)) − foto|`, com pixels em [-1, 1] | Menor: o conteúdo foi preservado |
| `mudanca_l1` | Média de `|G(foto) − foto|` | Nem 0 (não mudou nada) nem muito alto (destruiu a foto) |
| `nn_dist_fake_mean` | Distância L2 média de cada G(foto) até a pintura mais parecida | Perto da referência abaixo |
| `nn_dist_ref_mean` | Distância L2 média de cada pintura real até a mais parecida entre as **outras** | É a referência: o quanto duas pinturas diferentes de Van Gogh se parecem |
| `nn_dist_fake_min` | A menor distância entre uma saída e uma pintura | Muito perto de 0 indica cópia |

### FID (*Fréchet Inception Distance*)

Passa as imagens por uma rede já treinada (Inception v3) e pega 2.048 números que descrevem cada imagem. Depois, compara a **média** e a **covariância** desses números nos dois conjuntos. Quanto menor, mais parecidas as duas "nuvens" de imagens.

- **Limitação aqui:** o FID precisa de muitas imagens para estimar a covariância de 2.048 dimensões. Com 400 pinturas, ele é instável e enviesado para cima. Use para comparar modelos entre si, não como número absoluto.

### KID (*Kernel Inception Distance*)

Usa as mesmas features do FID, mas compara os conjuntos com uma função de kernel. Ele é calculado em vários subconjuntos de 100 imagens, o que dá também um desvio padrão.

- **Por que usar:** não tem o viés do FID com amostras pequenas, e o desvio padrão mostra se a diferença entre dois modelos é real ou ruído.

### O que nenhuma métrica captura

- O FID e o KID medem **estilo**, não preservação do conteúdo. Um gerador que ignora a foto e devolve uma pintura qualquer pode ter FID **melhor**. Por isso o `cycle_l1` vem junto.
- Nenhuma mede se o **céu** ficou bom. A máscara e a composição são avaliadas visualmente, com exemplos.
- A distância em pixels para memorização só detecta cópias quase literais, não cópias de composição ou de pedaços.

## Figuras: `report/figuras/`

| Arquivo | Conteúdo |
|---|---|
| `<experimento>_curvas.png` | Losses dos geradores e dos discriminadores (média móvel) |
| `<experimento>_amostras.png` | 8 fotos de teste: foto, G(foto) e F(G(foto)) |
| `<experimento>_piores.png` | As 8 fotos com maior erro de ida e volta |
| `<experimento>_vizinhos.png` | As 8 saídas mais próximas de uma pintura, ao lado dessa pintura |
| `<experimento>_memorizacao.png` | Histograma: distância saída → pintura x distância entre pinturas reais |
| `../tabela_comparativa.md` | Tabela com todos os experimentos avaliados |
