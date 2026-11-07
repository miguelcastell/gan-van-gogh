# ADR 0008: Treinar só com fotos de céu marcado

- **Status:** Aceito
- **Data:** 2026-11-03

## Contexto

Os treinos 04 e 05 mostraram que aumentar a resolução e a capacidade do gerador muda pouco. No teste em alta resolução com a Estátua da Liberdade ([experimento do treino 05](../experimentos/2026-11-02-nove-blocos.md#teste-em-alta-resolução-0311)), o céu cinza e liso virou uma textura granulada, sem pinceladas. Os céus que ficam bonitos são os que já têm estrutura: pôr do sol, nuvens marcadas.

Até agora, o gerador treinava com 2.000 fotos sorteadas do `trainB`, entre florestas, rios e cidades, muitas com pouco ou nenhum céu. No pipeline final só o céu é pintado, então boa parte do que o gerador aprendia não era usada.

## Decisão

Treinar a CycleGAN só com as fotos de `trainB` que têm céu "interessante", separadas por [`src/selecionar_fotos.py`](../../src/selecionar_fotos.py):

1. **Fração de céu** ≥ 25%, pela probabilidade da classe "céu" do SegFormer-B0 (o mesmo segmentador do `stylize`). Passaram 4.090 das 6.287 fotos.
2. Entre essas, uma **nota** = *z-score* do colorido do céu (métrica de Hasler e Süsstrunk, 2003, alta em pôr do sol) + *z-score* do desvio padrão da luminância no céu (alto com nuvens marcadas).
3. Ficam as **1.500** de maior nota. A lista está em `data/selecao_ceu/selecionadas.txt` e é passada ao treino com `--photos-list`.

As 400 pinturas continuam todas.

## Alternativas consideradas

- **Filtrar as pinturas com céu:** já descartada em [03-proximos-passos](../guia/03-proximos-passos.md). Reduziria as 400 pinturas para ~200, e o discriminador decoraria a coleção. Aqui o filtro é nas fotos, que são muitas (6.287).
- **Treinar mais tempo com as fotos de sempre (200 épocas, como no artigo):** levaria ~2 dias de GPU e não muda o que o gerador vê.
- **Separar as fotos à mão:** mais preciso, mas inviável para 6.287 imagens. A [folha de conferência](../../data/selecao_ceu/amostra.png) (as 64 primeiras e as 32 últimas que entraram) mostra que o critério automático funciona: pores do sol, nuvens dramáticas, céus com gradiente.

## Consequências

- O gerador concentra a capacidade em "céu de foto → céu de Van Gogh", que é o que o pipeline usa.
- O gerador fica pior em fotos sem céu (florestas, interiores), o que não importa para o pipeline, mas importa para pintar a foto inteira (`--sem-mascara`).
- A avaliação continua nas 751 fotos de `testB` (todas), para ser comparável com os treinos anteriores. Isso tende a desfavorecer o treino 06 no FID, porque metade do `testB` tem pouco céu.
- Uma época passa a ter 1.500 passos (eram 2.000).

## Resultado (2026-11-04)

O [treino 06](../experimentos/2026-11-03-fotos-de-ceu.md) não mostrou ganho claro: FID 127,1 contra 123,5 do treino 04, e o céu ficou mais saturado no pôr do sol, mas pior nas nuvens e no sol. A decisão continua registrada como foi tomada; qual modelo usar na entrega fica para depois da comparação com o `src.stylize`.

**Atualização (2026-11-04):** ao ver as imagens do céu, o autor achou o treino 06 claramente melhor e o escolheu como modelo da entrega ([ADR 0010](0010-treino-06-como-modelo-da-entrega.md)).
