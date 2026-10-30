# Experimento: primeiro treino, CycleGAN completa x sem *cycle loss*

- **Data:** 2026-10-29 (início às ~22h30, durante a noite)
- **Commit do código:** `0247887`
- **Comandos:**
  ```powershell
  python -m src.train_cyclegan --epochs 40 --photos 2000
  python -m src.train_cyclegan --epochs 40 --photos 2000 --sem-cycle
  python -m src.evaluate --model cyclegan
  python -m src.evaluate --model cyclegan_sem_cycle
  python -m src.plots
  ```
- **Pastas de saída:** `outputs/cyclegan/`, `outputs/cyclegan_sem_cycle/`, `report/figuras/`

## Pergunta

1. Com o orçamento que cabe numa noite (40 épocas de 2.000 passos em 128x128), a CycleGAN consegue levar fotos para o estilo de Van Gogh?
2. O que acontece quando se tira a *cycle loss*?

## Previsão (antes de rodar)

Nas palavras do autor, escritas antes do treino:

> "Eu espero que o primeiro treino não fique tão bom comparado ao original, mas tudo bem, por ser o primeiro é normal."

"Original" = os resultados do artigo da CycleGAN, que usa 256x256 e 200 épocas com todas as fotos. Este treino usa 1/4 dos pixels e cerca de 1/16 dos passos.

## Configuração

Só o que difere dos padrões da [ficha de hiperparâmetros](../referencia/hiperparametros.md):

| Parâmetro | Completa | Sem *cycle loss* |
|---|---|---|
| Épocas / fotos por época | 40 / 2.000 (80.000 passos) | 40 / 2.000 (80.000 passos) |
| λ_cycle / λ_identity | 10 / 5 | **0** / 5 |
| Queda do lr | a partir da época 20 | a partir da época 20 |
| Seed | 42 | 42 |

## Dados

- **Treino:** as 400 pinturas de `trainA` e 2.000 fotos de `trainB`, sorteadas com a seed 42 (as mesmas nos dois treinos).
- **Grades por época:** as 8 primeiras fotos de `testB`.
- **Avaliação:** as 751 fotos de `testB`, contra as 400 pinturas ([ficha dos dados](../referencia/dados.md)).

## Diário da execução

Atualizado durante a noite. Logs brutos em [`outputs/primeiro_treino/`](../../outputs/primeiro_treino/): a saída do terminal de cada passo (`*.log`), a GPU minuto a minuto (`gpu.csv`) e o tempo e o código de saída de cada passo (`resumo.json`). Os passos rodam em sequência, sem intervenção, por um script que impede o Windows de suspender.

| Hora | Evento |
|---|---|
| 22:21 | Início do treino completo. GPU a 51 °C, 1.927 MHz (máx. 2.100), 1,6 GB de VRAM, 100% de uso |
| 22:22 | Primeiros 100 passos: ~173 ms por passo, estimativa de 5,8 min por época (~4 h por treino, ~8 h no total). Os 130 ms medidos antes eram com imagens falsas e sem o tempo de aquecimento |
| 22:30 | Época 1 concluída em **8,7 min** (~260 ms por passo), não nos 5,8 previstos. A GPU está saudável: 71 a 73 °C, ~1.912 MHz, 96 a 99% de uso, ~72 W, sem redução de clock por temperatura. Losses no fim da época: G_adv 0,245, cycle 0,384, D_pintura 0,380 |
| 22:47 | Época 5 concluída (1º checkpoint salvo). As épocas 2 a 5 levaram **4,2 min cada** (~126 ms por passo): só a 1ª foi lenta, pelo aquecimento (o `cudnn.benchmark` testando os algoritmos e a primeira carga dos dados). Estimativa corrigida: ~2,9 h por treino, fim de tudo por volta das 4h30. GPU a 74 °C e 100% de uso. As losses são de **um único passo** (batch 1) e oscilam muito: D_pintura foi a 0,034 na época 2 e voltou a 0,242 na 5. A tendência só aparece na média móvel do `logs.csv` |
| 23:08 | Época 10 concluída (2º checkpoint). Média por época no `logs.csv`: a *cycle loss* caiu de 0,609 (época 1) para 0,299 (época 10), e a *identity*, de 0,568 para 0,274. Os discriminadores estão abaixo de 0,25 (o valor de quem "chuta") e descendo: D_pintura de 0,399 para 0,151, D_foto de 0,331 para 0,180. Ou seja, os D estão ganhando o jogo aos poucos. No começo da época 11, o D_pintura caiu para 0,075 e o G_adv subiu para 0,70: **ponto de atenção**, porque pode ser o D_pintura decorando as 400 pinturas |
| 23:08 | **Grade da época 10** ([`epoch_010.png`](../../outputs/cyclegan/samples/epoch_010.png)): o estilo já apareceu. Paleta amarelo, verde e ocre, textura de pincelada visível, e os céus do templo e da praia com traços em movimento. A volta (linha 3) reconstrói as fotos bem, com cores um pouco lavadas. Primeiro defeito claro: o **pato branco** quase desaparece na pintura, se misturando à textura da água |
| 23:29 | Época 15 concluída (3º checkpoint). **Platô:** da época 10 à 15, a *cycle loss* média ficou entre 0,285 e 0,310, e a *identity*, entre 0,251 e 0,274. Os discriminadores estabilizaram (D_pintura entre 0,120 e 0,151, D_foto entre 0,107 e 0,189): o D_pintura **não** continuou caindo depois do alerta da época 11, então não houve colapso. A queda do lr começa na época 21 e deve ajudar a sair do platô |
| 23:29 | **Grade da época 15** ([`epoch_015.png`](../../outputs/cyclegan/samples/epoch_015.png)): qualidade parecida com a da época 10, sem melhora visível. Dois fenômenos para analisar: (1) o pato branco continua **sumido na pintura, mas reaparece na volta** (linha 3). Hipótese levantada na hora: o gerador estaria "escondendo" a informação do pato em sinais quase invisíveis, um efeito conhecido da CycleGAN chamado de **esteganografia** (Chu et al., 2017, *CycleGAN, a Master of Steganography*). **Testada depois e não confirmada** (veja "O que observei"). (2) Na última foto aparece um **artefato quadrado em volta do sol**, na pintura e na volta. Áreas muito claras e saturadas parecem difíceis para o gerador |
| 23:50 | Época 20 concluída (4º checkpoint), fim da fase de lr constante. A *cycle loss* média voltou a cair devagar (0,285 na época 15, 0,259 na 20), e a *identity*, de 0,251 para 0,231. O D_pintura ficou entre 0,080 e 0,120: forte, mas estável. Detalhe da agenda: com `decay_start` = 20, o lr ainda é 2e-4 na época 21 e só começa a cair a partir da 22 (até 1e-5 na 40) |
| 23:50 | **Grade da época 20** ([`epoch_020.png`](../../outputs/cyclegan/samples/epoch_020.png)): o **pato branco agora aparece na pintura**, e o artefato no sol sumiu. Defeito novo: no céu roxo e liso da 5ª foto (a casinha) surgiram **listras verticais e um padrão de grade**, na pintura e na volta. Esse tipo de padrão periódico é típico das convoluções transpostas do decodificador (*checkerboard artifacts*, Odena et al., 2016) e aparece mais em áreas lisas, que no projeto são justamente os céus |
| 00:10 | Época 25 concluída (5º checkpoint). O lr está caindo como previsto (1,9e-4 na época 22, 1,6e-4 na 25). A *cycle loss* média continua descendo devagar (0,259 na 20, 0,222 na 25), e a *identity*, de 0,231 para 0,185. O D_pintura ficou mais forte (entre 0,060 e 0,093) e o D_foto, estável (entre 0,126 e 0,168). As médias por época oscilam bastante: o F_adv, por exemplo, foi a 1,0 no começo da época 26 |
| 00:10 | **Grade da época 25** ([`epoch_025.png`](../../outputs/cyclegan/samples/epoch_025.png)): pinceladas mais marcadas e com mais contraste, principalmente na praia e no templo, e a volta cada vez mais fiel. As listras no céu da casinha diminuíram, mas não sumiram. O pato voltou a ficar apagado na pintura: a cada época, o gerador alterna entre mostrar e esconder detalhes pequenos e claros |
| 00:31 | Época 30 concluída (6º checkpoint). Lr em ~1,1e-4. A *cycle loss* média ficou entre 0,222 e 0,256 nas épocas 25 a 30, ou seja, **voltou a estagnar**, apesar do lr menor. Os discriminadores continuam estáveis (D_pintura entre 0,069 e 0,100, D_foto entre 0,134 e 0,179). Não há sinal de colapso: o jogo está equilibrado, com os D um pouco à frente |
| 00:31 | **Grade da época 30** ([`epoch_030.png`](../../outputs/cyclegan/samples/epoch_030.png)): a pincelada ficou mais densa e direcional (traços curtos e paralelos, bem característicos de Van Gogh), principalmente na grama, na água e no céu da casinha. Ali as listras de grade viraram traços verticais que parecem pinceladas intencionais. A reconstrução continua boa. A melhora visual entre as épocas 25 e 30 é sutil |
| 00:53 | Época 35 concluída (7º checkpoint). Lr em ~6e-5. **Agora a queda do lr fez efeito:** a *cycle loss* média caiu de 0,245 (época 30) para 0,208 (35) e 0,186 (começo da 36), e a *identity*, de 0,201 para 0,164. O D_pintura também ficou mais forte (de 0,073 para 0,046). Passos menores deixam os dois lados refinarem o que já sabem, em vez de ficarem "pulando" em volta da solução |
| 00:53 | **Grade da época 35** ([`epoch_035.png`](../../outputs/cyclegan/samples/epoch_035.png)): pinceladas mais longas e expressivas, principalmente no parque, na praia e no campo dourado. O céu da casinha virou uma textura de traços azuis e verdes, sem o padrão de grade da época 20. Efeito colateral: o **templo ficou esverdeado**, porque o gerador puxa a paleta inteira para os verdes e amarelos de Van Gogh, mesmo onde a foto não tem essas cores |
| 01:14 | **Treino completo concluído** em 173 min (código de saída 0). 8 checkpoints (épocas 5 a 40, 1,9 GB no total, fora do git) e `evolucao.gif` (12 MB). Médias da última época: *cycle* 0,207, *identity* 0,161, G_adv 0,913, F_adv 0,453, D_pintura 0,030, D_foto 0,161. Na reta final, o **D_pintura foi ficando cada vez mais forte** (de 0,046 na época 35 para 0,030 na 40) e o G_adv subiu (de 0,79 para 0,91): o discriminador de pinturas terminou à frente do gerador |
| 01:14 | **Grade da época 40** ([`epoch_040.png`](../../outputs/cyclegan/samples/epoch_040.png)): o resultado mais "pintado" da noite, com pinceladas longas e marcadas e paleta de Van Gogh em todas as fotos. A volta (linha 3) reconstrói bem todas as cenas, inclusive o pato. Continuam: o templo esverdeado e o pato branco apagado na pintura |
| 01:14 | Início do treino **sem *cycle loss*** (λ_cycle = 0, λ_identity = 5). Primeiros 100 passos: ~137 ms por passo. Ele roda o gerador 2 vezes a menos por passo, porque não precisa calcular a volta |
| 01:35 | Sem *cycle loss*, época 5 (1º checkpoint). **Todas as épocas levaram 4,2 min**, o mesmo tempo do treino completo, mesmo com 2 execuções do gerador a menos por passo (e sem a época inicial mais lenta). Isso sugere que o gargalo não é só a conta na GPU: parte do tempo vai para o Python, a cópia dos batches para a GPU e o *image pool*, que são iguais nos dois treinos. Vale investigar antes de treinos maiores |
| 01:35 | Médias da época 5: a *cycle loss* (que aqui é só **medida**, não otimizada) ficou em 0,634, contra 0,354 do treino completo na mesma época. Os discriminadores estão perto de 0,25 (D_pintura 0,272, D_foto 0,245), ou seja, mais perto do "chute" do que no treino completo, onde já estavam em ~0,18 |
| 01:35 | **Grade da época 5** ([`epoch_005.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_005.png)): **surpresa em relação à previsão da ADR 0002.** O conteúdo das fotos **foi preservado** (o pato, o templo e a casinha continuam lá), mas a estilização é bem mais fraca: parece a foto com um filtro levemente pintado, sem as pinceladas do treino completo. Já a volta (linha 3) é claramente pior, com padrão de grade e cores erradas. Hipótese: o que segurou o conteúdo aqui foi a ***identity loss***, que continuou ligada, junto com a arquitetura residual, que facilita "não mudar nada". Sem a *cycle loss*, o gerador ficou no caminho mais fácil: mudar pouco a foto |
| 01:56 | Sem *cycle loss*, época 10 (2º checkpoint). A *cycle loss* medida **piorou** (de 0,634 na época 5 para 0,658 na 10 e 0,791 no começo da 11), enquanto no treino completo ela caía. A *identity* continua caindo (de 0,315 para 0,235). Os discriminadores ficaram perto de 0,25 (D_pintura entre 0,225 e 0,272): sem ter que preservar a foto, os geradores enganam os D com mais facilidade |
| 01:56 | **Grade da época 10** ([`epoch_010.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_010.png)): **agora a previsão da ADR 0002 se confirma.** Entre as épocas 5 e 10, o gerador passou a **inventar conteúdo**: a praia roxa virou uma paisagem verde, o céu da casinha virou uma massa de árvores, o templo ficou verde e quase se dissolve no fundo, e os patos mudaram de cor (o branco ficou amarelo, o preto ficou vermelho). A volta (linha 3) **perdeu a cena** em 5 das 8 fotos: no lugar delas aparecem céus azuis genéricos e borrões. Isso corrige a hipótese da época 5: a *identity loss* só segurou o conteúdo no começo, enquanto o gerador ainda estava perto da identidade. Com o treino, nada o impede de trocar a foto por "qualquer coisa que pareça Van Gogh". Esse é exatamente o papel da *cycle loss* |
| 02:17 | Sem *cycle loss*, época 15 (3º checkpoint). A *cycle loss* medida ficou entre 0,654 e 0,685 (contra ~0,29 no treino completo). Agora o **D_pintura voltou a ganhar** (de 0,248 na época 10 para 0,146 na 15 e 0,083 no começo da 16), e o G_adv subiu (de 0,27 para 0,39): o mesmo padrão do treino completo, só que atrasado |
| 02:17 | **Grade da época 15** ([`epoch_015.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_015.png)): o resultado **oscila**. Algumas fotos voltaram a ser reconhecíveis (a praia, o templo, a cidade), com céus verde-água bem ao estilo de Van Gogh. Outras continuam inventadas: a casinha sumiu e virou uma massa escura de árvores, e o pato branco se dissolveu na água. A volta está escura e perdeu a casinha, o pato e o campo. Sem a *cycle loss*, nada "ancora" o gerador numa versão da foto, e ele muda de solução de uma época para outra |
| 02:38 | Sem *cycle loss*, época 20 (4º checkpoint). A *cycle loss* medida ficou entre 0,616 e 0,715 nas épocas 15 a 20. O D_pintura continua forte (entre 0,104 e 0,133), e o D_foto, perto de 0,22. Começa agora a queda do lr |
| 02:38 | **Grade da época 20** ([`epoch_020.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_020.png)): é a grade **mais "Van Gogh" da noite inteira**, até mais que a do treino completo, com pinceladas grossas, amarelos e azuis saturados e textura em tudo. Mas o conteúdo foi sacrificado: a praia virou uma mancha abstrata, a casinha não existe mais, o templo virou uma silhueta escura, e o céu do campo dourado ficou azul-escuro. A volta é irreconhecível em quase todas as fotos. É o *trade-off* que a comparação queria mostrar: **sem a *cycle loss*, o estilo fica mais forte justamente porque o gerador para de respeitar a foto**. Previsão para a avaliação: o FID/KID desta versão pode sair **igual ou melhor** que o da completa, enquanto a `cycle_l1` sai muito pior |
| 02:59 | Sem *cycle loss*, época 25 (5º checkpoint). A *cycle loss* medida subiu para 0,708 a 0,775. O D_pintura está forte e estável (entre 0,068 e 0,090) |
| 02:59 | **Grade da época 25** ([`epoch_025.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_025.png)): **sinal de colapso de modo parcial.** As 8 pinturas passaram a ter a **mesma "assinatura"**: uma mancha rosa e vermelha no canto de cima e a mesma paleta verde e amarela, seja qual for a foto. A volta ficou ainda mais uniforme, e quase todas viraram a mesma paisagem de "montanhas laranja" sobre fundo escuro. Ou seja, o F aprendeu a devolver uma "foto genérica" que engana o D_foto, sem relação com a entrada. É o "truque" que o guia 01 descreve: aplicar a mesma textura em qualquer foto, ignorando o que há nela |
| 03:20 | Sem *cycle loss*, época 30 (6º checkpoint). A *cycle loss* medida chegou a 0,694 a 0,848 (e 0,933 no começo da 31). D_pintura entre 0,061 e 0,090 |
| 03:20 | **Grade da época 30** ([`epoch_030.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_030.png)): **colapso de modo claro.** Cinco das oito fotos (a praia, a bola de cristal, a casinha, os patos e o templo) viraram **quase a mesma imagem**: uma moldura azul e verde com textura de grade, onde só se vê um "fantasma" da foto original. Na volta, as cinco viraram a mesma paisagem de montanhas azuis. Curiosamente, o parque (1ª foto) saiu **quase sem mudança nenhuma**, e a cidade e o campo dourado ainda têm cenas próprias. O gerador está tratando entradas diferentes de jeitos totalmente diferentes, sem uma regra consistente. É o exemplo mais claro da noite do que a *cycle loss* evita |
| 03:41 | Sem *cycle loss*, época 35 (7º checkpoint). A *cycle loss* medida subiu para 0,885 (época 35) e 1,254 no começo da 36, o maior valor da noite. A queda do lr **não** ajudou aqui: ela só refina o que o modelo já faz, e o que ele faz é colapsar. D_pintura entre 0,063 e 0,099 |
| 03:41 | **Grade da época 35** ([`epoch_035.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_035.png)): **colapso de modo quase total.** Seis das oito fotos viraram **a mesma imagem**: uma composição em tons pastel com um "prédio" no alto, que não existe em nenhuma delas. A cidade virou uma mancha amarela com um "sol" no meio. Só o parque continua reconhecível. Na volta, as seis viraram a mesma cena esbranquiçada. O gerador encontrou uma única saída que engana o D_pintura e passou a usá-la para quase qualquer entrada |
| 04:02 | **Treino sem *cycle loss* concluído** em 168 min (código de saída 0). Médias da última época: *cycle* (medida) 0,896, *identity* 0,127, G_adv 0,536, F_adv 0,273, D_pintura 0,067, D_foto 0,221 |
| 04:02 | **Grade da época 40** ([`epoch_040.png`](../../outputs/cyclegan_sem_cycle/samples/epoch_040.png)): **colapso total.** Sete das oito fotos viraram a **mesma paisagem verde com uma construção ao fundo**, praticamente idênticas entre si. Só o parque continua sendo o parque. A volta também ficou igual para as sete. Os números do log não denunciam isso: a G_adv e a D_pintura parecem "saudáveis". Só a *cycle loss* medida (0,9 contra 0,2) e **olhar as imagens** revelam o colapso. Lição para o artigo: **as losses de uma GAN não bastam para julgar o resultado** |
| 04:02 | Início da avaliação (`evaluate` nos dois modelos) e dos gráficos (`plots`) |

## Resultado

**Tempo:** o treino completo levou 173 min e o sem *cycle loss*, 168 min (~126 ms por passo depois da 1ª época, nos dois). A avaliação e os gráficos levaram cerca de 1 min. Tudo terminou às 04h03, sem erros ([`resumo.json`](../../outputs/primeiro_treino/resumo.json)). A GPU ficou entre 70 e 76 °C, a ~1.912 MHz, sem redução de clock por temperatura ([`gpu.csv`](../../outputs/primeiro_treino/gpu.csv)).

**Métricas finais** (checkpoint da época 40, 751 fotos de teste, contra as 400 pinturas; [`metrics.json` da completa](../../outputs/cyclegan/metrics.json), [`metrics.json` da sem *cycle*](../../outputs/cyclegan_sem_cycle/metrics.json)):

| | Completa | Sem *cycle loss* | Referência |
|---|---|---|---|
| FID ↓ | **125,3** | 220,9 | 179,8 (fotos sem conversão) |
| KID ↓ | **0,051 ± 0,005** | 0,190 ± 0,015 | 0,100 (fotos sem conversão) |
| Diversidade das saídas | **128,9** | 56,9 | 137,7 (pinturas reais) |
| Erro de ida e volta (`cycle_l1`) ↓ | **0,130** | 0,494 | — |
| Mudança foto → pintura (`mudanca_l1`) | 0,295 | 0,489 | — |
| Distância à pintura mais próxima (média / mínima) | 93,1 / 58,2 | 90,7 / 69,8 | 93,2 (entre pinturas reais) |

**Por checkpoint** (calculado depois, com o mesmo método; [`por_checkpoint.csv`](../../outputs/primeiro_treino/por_checkpoint.csv)). O KID usa subconjuntos sorteados, então varia ~0,001 entre execuções:

| Época | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
|---|---|---|---|---|---|---|---|---|
| FID completa | 168,2 | 137,8 | 144,5 | 136,9 | **124,2** | 130,9 | 128,2 | 125,3 |
| FID sem *cycle* | 166,8 | 151,1 | **140,5** | 172,7 | 154,9 | 177,9 | 230,4 | 220,9 |
| Diversidade completa | 130,5 | 122,0 | 131,3 | 137,8 | 131,6 | 130,4 | 128,5 | 128,9 |
| Diversidade sem *cycle* | 128,4 | 118,1 | 124,6 | 117,2 | 99,3 | 88,1 | 56,0 | 56,9 |

**Figuras:** [curvas da completa](../../report/figuras/cyclegan_curvas.png), [curvas sem *cycle*](../../report/figuras/cyclegan_sem_cycle_curvas.png), [piores casos](../../report/figuras/cyclegan_piores.png), [vizinhos](../../report/figuras/cyclegan_vizinhos.png), [memorização](../../report/figuras/cyclegan_memorizacao.png), [tabela comparativa](../../report/tabela_comparativa.md). Teste do céu em fotos de teste: [templo](../../outputs/stylize/2014-08-03%2009_47_19_cyclegan_painel.png), [casinha](../../outputs/stylize/2014-08-02%2015_56_41_cyclegan_painel.png), [praia](../../outputs/stylize/2014-08-01%2022_38_22_cyclegan_painel.png).

A análise completa, com gráficos e testes que dá para rodar, está no notebook [`03_analise_primeiro_treino.ipynb`](../../notebooks/03_analise_primeiro_treino.ipynb).

## O que observei

1. **A CycleGAN completa funcionou.** O FID caiu de 179,8 (foto sem conversão) para 125,3, e a diversidade das saídas (128,9) ficou perto da diversidade das pinturas reais (137,7): não houve colapso. O estilo apareceu já na época 10.
2. **O treino completo estagnou por volta da época 25.** Depois disso, o FID oscilou entre 124 e 131, e a melhora visual foi pequena. Para este orçamento, 25 a 30 épocas bastariam.
3. **Sem a *cycle loss*, o modelo colapsou.** Até a época 15 ele estava até um pouco melhor no FID (140,5 contra 144,5) e preservava razoavelmente o conteúdo. Da época 20 em diante, a diversidade despencou (117 → 57), e na época 40, 7 das 8 fotos fixas viravam a mesma imagem. A ADR 0002 previa a perda de conteúdo, mas não o colapso.
4. **A previsão "FID igual ou melhor sem a *cycle loss*" (escrita às 02h38) acertou até a época 15 e errou no fim.** O FID não mede só "parece Van Gogh?": ele também compara a **variedade** das imagens. Quando o modelo colapsa, o FID pune.
5. **As losses não denunciaram o colapso.** No treino sem *cycle*, a G_adv e a D_pintura pareciam normais. Só a *cycle loss* medida, a diversidade e as próprias imagens mostraram o problema.
6. **O D_pintura terminou dominando nos dois treinos** (~0,03 a 0,07, longe do 0,25 de quem chuta), e a G_adv subiu na segunda metade. Com 400 pinturas vistas ~5 vezes por época, o discriminador de pinturas tem facilidade em decorá-las.
7. **Não houve memorização** detectável em pixels: a distância das saídas à pintura mais próxima (93,1) é igual à de uma pintura real até outra (93,2). Quase todos os "vizinhos" são a mesma pintura (a *Cadeira de Gauguin*), que fica no meio do espaço de pixels: é uma limitação da métrica, não uma cópia.
8. **Os piores casos são céus noturnos e céus lisos de pôr do sol** (maior erro de ida e volta): a Via Láctea, um céu estrelado, um sol vermelho sobre fundo liso. Isso está fora do que o modelo viu (paisagens diurnas), e é justamente o tipo de céu que o projeto quer pintar.
9. **Esteganografia: hipótese não confirmada.** Somar ruído (σ até 0,1) à pintura antes da volta quase não mudou o erro de reconstrução (0,130 → 0,135). Se o pato estivesse escondido num sinal fraco, o ruído o destruiria. O mais provável é que ele continue na pintura com pouco contraste.
10. **O céu de Van Gogh funcionou de ponta a ponta:** no templo, o céu saiu pintado e o templo, intacto. No céu liso da casinha apareceram moldura e grade. Rodando o gerador em 128 (a escala do treino) em vez de 256 (o padrão do `stylize`), os artefatos somem, mas o resultado fica mais borrado.
11. **Desempenho:** sem *cycle loss*, o gerador roda 2 vezes a menos por passo, e mesmo assim cada época levou os mesmos 4,2 min. Então o gargalo não está só na GPU.

## Conclusão e próximos passos

**Sobre a previsão do autor:** confirmada. O resultado é visivelmente Van Gogh, mas não chega ao nível do artigo original: tem menos resolução, pinceladas mais grossas e artefatos em céus lisos.

**Sobre a pergunta do experimento:** a *cycle loss* é o que mantém o gerador honesto. Sem ela, o modelo primeiro inventa conteúdo e depois colapsa numa saída única.

**O que ajustar no próximo treino**, em ordem de impacto esperado:

1. **Recorte aleatório (*random crop*) no aumento de dados:** o artigo redimensiona para 286 e recorta 256. Nós só espelhamos. Recortes diferentes da mesma pintura dificultam o D_pintura decorar as 400 pinturas (observação 6).
2. **Treinar em 256x256** para pinceladas mais finas e para o `stylize` não precisar rodar fora da escala do treino (observação 10). Custo: ~4 vezes mais tempo por passo.
3. **Trocar as convoluções transpostas por *upsample* + convolução** no decodificador, a correção clássica para os padrões de grade (Odena et al., 2016).
4. **Menos épocas:** 25 a 30 bastaram neste orçamento (observação 2). O tempo economizado pode ir para os itens 1 e 2.
5. **Investigar o gargalo de desempenho** antes de treinos maiores (observação 11).
6. **Céus noturnos:** decidir se ficam fora do escopo (e entram como limitação no artigo) ou se vale incluir fotos noturnas no treino (observação 8).
