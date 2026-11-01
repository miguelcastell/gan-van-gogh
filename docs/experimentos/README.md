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
