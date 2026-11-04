# ADR 0009: Transferência de estilo neural como comparação

- **Status:** Aceito
- **Data:** 2026-11-03

## Contexto

O autor esperava ver "uma Noite Estrelada no fundo" das fotos. A CycleGAN não faz isso: ela aprende o estilo médio das 400 pinturas (a maioria campos, retratos e naturezas-mortas), e as perdas de ciclo e de identidade a obrigam a preservar a foto. Por isso ela troca cor e textura, mas não cria redemoinhos. Essa limitação já estava prevista no [plano](../../projeto-final-gan.md#2-a-ideia) e apareceu claramente no teste com a Estátua da Liberdade.

Restrições: entrega em 02/12/2026, uma RTX 3050 de 6 GB, pouco tempo. O enunciado pede uma GAN, então a CycleGAN continua sendo o centro do projeto.

## Decisão

Adicionar a **transferência de estilo neural de Gatys, Ecker e Bethge (2015)** ([`src/transferencia_estilo.py`](../../src/transferencia_estilo.py)) com a *Noite Estrelada* como estilo, **só como comparação** na apresentação. Ela usa o mesmo pipeline de máscara do céu (`--so-ceu`).

Configuração: VGG19 pré-treinada do torchvision, conteúdo em conv4_2, estilo (matrizes de Gram) em conv1_1 a conv5_1, peso de estilo 10⁶, L-BFGS com 300 iterações a partir da foto, imagem com 768 px no maior lado. Cerca de 1 min por imagem na RTX 3050.

## Alternativas consideradas

- **AdaIN (Huang e Belongie, 2017):** gera em uma passada, mas precisa de um decodificador treinado, que o torchvision não traz. Seria mais um modelo para baixar ou treinar.
- **Não comparar e só documentar a limitação:** mais simples, mas o resultado visual da entrega ficaria fraco, e a comparação rende uma boa discussão para o relatório.
- **Trocar a CycleGAN pela transferência neural:** fugiria do enunciado (GAN).

## Consequências

- A apresentação ganha uma comparação clara: **a GAN aprende o estilo de um pintor a partir de 400 quadros e converte qualquer foto numa passada; a transferência neural copia um único quadro, e cada foto exige uma otimização de ~1 min.**
- A transferência neural não é uma GAN e não entra nas métricas de GAN (FID por checkpoint, curvas de G e D). É uma referência visual.
- O download da VGG19 (548 MB) fica no cache do torch, fora do repositório.
