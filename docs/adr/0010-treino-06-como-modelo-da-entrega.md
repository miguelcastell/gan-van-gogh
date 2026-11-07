# ADR 0010: Treino 06 como modelo da entrega

- **Status:** Aceito
- **Data:** 2026-11-04

## Contexto

Depois de seis treinos, três deles em 256x256 (04, 05 e 06) empataram nas métricas: FID 123,5 (treino 04, época 30), 122,3 (treino 05, época 30) e 127,1 (treino 06, época 40). As diferenças estão dentro da oscilação de ~4 pontos entre checkpoints vizinhos, então o FID não decide qual usar. E o FID não mede o que o pipeline entrega, que é só o céu ([ficha das métricas](../referencia/metricas.md#o-que-nenhuma-métrica-captura)).

O [registro do treino 06](../experimentos/2026-11-03-fotos-de-ceu.md) sugeria manter o treino 04, com base nas observações do Claude sobre as cores das nuvens e o sol apagado.

## Decisão

O modelo da entrega é o **treino 06** (`cyclegan_256_ceu`, checkpoint `epoch_040.pt`). A escolha foi do autor, ao ver as imagens do céu em 2026-11-04: "curti muito os resultados, ficou muito top, realmente melhorou bastante".

## Alternativas consideradas

- **Treino 04 (`cyclegan_256`, época 30):** melhor FID dos de 256 com 6 blocos e cores de nuvem mais próximas da paleta de Van Gogh. Não foi escolhido porque, na avaliação visual do autor, o céu do treino 06 parece mais pintado, que é o objetivo do projeto.
- **Treino 05 (`cyclegan_256_9blocos`, época 30):** menor FID (122,3), mas o treino foi encerrado na época 30 e a melhora sobre o 04 não aparece nas imagens.

## Consequências

- O modelo escolhido é coerente com o pipeline: foi treinado justamente para céus ([ADR 0008](0008-treinar-com-fotos-de-ceu.md)).
- A escolha é visual e não pelo número. Nos slides, isso precisa ser dito, junto com o motivo: o FID mede a imagem inteira e não o céu.
- Os pontos fracos conhecidos vão junto: nuvens em tom verde-cáqui em algumas fotos, o disco do sol que some, céu liso que vira textura sem pinceladas, e fotos sem céu menos transformadas (não importa no pipeline, só com `--sem-mascara`).
- Comandos de demonstração passam a usar `--model cyclegan_256_ceu`.
