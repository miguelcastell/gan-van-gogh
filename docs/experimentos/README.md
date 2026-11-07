# Diário de experimentos

Um arquivo por experimento (treino, teste de hiperparâmetro, avaliação). Essas notas viram as seções de resultados e de "o que deu errado" do relatório.

## Como criar

1. Copie [`template.md`](template.md) para `AAAA-MM-DD-nome-curto.md` (ex.: `2026-10-30-cyclegan-1-epoca.md`).
2. Preencha a **previsão antes de rodar**. É o que transforma rodar código em aprender.
3. Adicione na tabela abaixo.

## Índice

| Data | Experimento | Pasta em `outputs/` | Conclusão em uma frase |
|---|---|---|---|
| 2026-10-29 | [Tempo por passo de treino](2026-10-29-tempo-por-passo.md) | — | ~130 ms por passo: 40 épocas com 2.000 fotos levam cerca de 3 horas |
| 2026-10-29 | [Primeiro treino: completa x sem cycle loss](2026-10-29-primeiro-treino.md) | `cyclegan`, `cyclegan_sem_cycle` | A completa chegou a FID 125 (fotos sem conversão: 180); sem *cycle loss*, o modelo colapsou (FID 221, diversidade 57 contra 138 das pinturas). Análise no notebook 03 |
| 2026-10-30 | [Recorte aleatório no aumento de dados](2026-10-30-recorte-aleatorio.md) | `cyclegan_crop` | Melhor modelo até agora: FID 117,4 e KID 0,036 na época 40, contra 125,3 e 0,051 do 1º treino; o D_pintura ficou menos dominante (0,097 contra 0,030) |
| 2026-10-31 | [Treinar em 256x256](2026-10-31-resolucao-256.md) | `cyclegan_256` | Melhor aparência até agora: detalhes preservados e primeiras pinceladas visíveis. Nas métricas, melhora pequena: FID −36% sobre as fotos sem conversão (−35% em 128); `cycle_l1` 0,111 (o menor do projeto). 8h18 de treino |
| 2026-11-02 | [256x256 com 9 blocos residuais (treino 05)](2026-11-02-nove-blocos.md) | `cyclegan_256_9blocos` | Encerrado na época 30 (de 40) a pedido do autor. FID 122,3, empatado com o treino 04 (123,5 na mesma época): mais blocos não ajudaram com 400 pinturas. Num céu cinza em alta resolução, só textura granulada, sem pinceladas |
| 2026-11-03 | [Transferência de estilo neural com a Noite Estrelada](2026-11-03-transferencia-neural.md) | `transferencia` | Na Estátua da Liberdade, pinceladas curvas da Noite Estrelada no céu cinza, onde a CycleGAN só fez textura granulada. ~1 min por imagem |
| 2026-11-03 | [Só fotos de céu marcado (treino 06)](2026-11-03-fotos-de-ceu.md) | `cyclegan_256_ceu` | FID 127,1 (época 40), empatado ou um pouco pior que o treino 04 (123,5). Céu mais saturado no pôr do sol, mas nuvens verde-cáqui e o sol apagado; sem ganho claro. 6h20 de treino |
| 2026-11-06 | [Treino 06 até a época 80, com λ_identity 5 e 1 (treinos 07 e 08)](2026-11-06-oitenta-epocas-e-identity.md) | `cyclegan_256_ceu_80`, `cyclegan_256_ceu_id1` | Treino 07: FID 119,2 na época 45 (menor de todos em 256) e ~125 até a 80; céu parecido com o do 06, um pouco mais quente e azul. Treino 08 encerrado por falta de RAM, pendente |
