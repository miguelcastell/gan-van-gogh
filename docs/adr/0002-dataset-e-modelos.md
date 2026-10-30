# ADR 0002: CycleGAN no vangogh2photo, com o estilo aplicado só no céu

- **Status:** Aceito
- **Data:** 2026-10-29

## Contexto

O professor sugeriu usar GAN para levar o traço de Van Gogh a fotos reais. O objetivo é tirar uma foto (por exemplo, de uma estátua) e receber a mesma foto com o **céu pintado no estilo de Van Gogh**, como na *Noite Estrelada*, e o resto intacto.

O enunciado pede uma GAN, uma comparação e avaliação de realismo, diversidade, estabilidade e exemplos ruins. Restrições: RTX 3050 com 6 GB de VRAM e entrega em 02/12/2026.

## Decisão

- **Dataset:** `vangogh2photo`, publicado com a CycleGAN: pinturas de Van Gogh (domínio A) e fotos de paisagens (domínio B), sem pares.
- **Modelo:** CycleGAN (Zhu et al., 2017), treinada por nós em 128x128.
- **Comparação:** CycleGAN completa x CycleGAN **sem a *cycle loss***, com o mesmo orçamento de treino. A *identity loss* continua ligada nas duas, para que a única diferença seja a *cycle loss*. A previsão é que, sem ela, o gerador fique livre para "inventar" uma pintura qualquer e perca o conteúdo da foto.
- **Céu:** um segmentador **pré-treinado** gera a máscara do céu. A foto estilizada é combinada com a original usando essa máscara, com a borda suavizada.

## Alternativas consideradas

- **Estilizar a imagem inteira, sem máscara:** mais simples, mas a estátua também ganharia pinceladas, e não é o resultado desejado.
- **Transferência de estilo neural (Gatys et al., 2015):** copiaria a *Noite Estrelada* em si (os redemoinhos e a lua), mas **não é uma GAN** e não atende o enunciado.
- **pix2pix:** é uma GAN de tradução de imagens, mas exige pares foto-pintura da mesma cena, que não existem.
- **DCGAN gerando pinturas do zero:** existem poucas centenas de pinturas de Van Gogh, pouco para treinar uma GAN do zero, e ela não recebe uma foto de entrada.
- **Comparar com e sem a *identity loss*:** também é válido, mas o efeito da *cycle loss* é mais visível e mais fácil de explicar.
- **Treinar o próprio segmentador de céu:** exigiria uma base com máscaras e tempo que não existe no prazo. Segmentar não é o foco do projeto.

## Consequências

- **Limitação do resultado:** a CycleGAN aprende o **estilo geral** de Van Gogh a partir de centenas de pinturas, não a *Noite Estrelada* especificamente. O céu terá as pinceladas e as cores dele, mas não os redemoinhos exatos daquele quadro. Isso precisa estar claro no relatório.
- **Treino pesado:** com a receita original (256x256, 200 épocas), o treino nessa GPU levaria dias. Com 128x128 e menos épocas, deve cair para horas. O número de épocas só será definido depois de medir o tempo de uma época.
- **Desequilíbrio entre domínios:** há bem menos pinturas do que fotos. O risco é o discriminador de pinturas decorar o treino.
- **Avaliação sem classes:** não há classes para medir diversidade como num dataset rotulado. Usaremos FID/KID contra as pinturas reais, a preservação do conteúdo (ida e volta) e o vizinho mais próximo para memorização.
- **Ética:** o tema traz a discussão sobre imitação de estilo artístico, autoria e manipulação de fotos. A obra de Van Gogh é de domínio público, mas a mesma técnica copia o estilo de artistas vivos.

## Em aberto

- [ ] Número de épocas e tamanho do subconjunto de fotos (depende do tempo medido no primeiro treino).
- [ ] Qual segmentador de céu usar. Ele precisa ter a classe "céu": modelos treinados no ADE20K têm; os do PASCAL VOC, como o DeepLab do torchvision, não. **Proposta:** SegFormer-B0 no ADE20K (`src/sky.py`), pequeno e rápido. Falta confirmar olhando as máscaras em fotos reais.
- [ ] Como medir diversidade sem classes.
