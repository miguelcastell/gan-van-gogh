# Experimento: transferência de estilo neural com a Noite Estrelada

- **Data:** 2026-11-03
- **Commit do código:** `d2532cc` + `src/transferencia_estilo.py` (ainda sem commit)
- **Comandos:**
  ```powershell
  python -m src.transferencia_estilo data/demo/liberdade.jpg --estilo data/estilos/noite_estrelada.jpg --max-side 768
  python -m src.transferencia_estilo data/demo/liberdade.jpg --estilo data/estilos/noite_estrelada.jpg --max-side 768 --so-ceu
  ```
- **Pasta de saída:** `outputs/transferencia/`

## Pergunta

Uma técnica que copia **um quadro específico** dá o resultado que o autor esperava ("uma Noite Estrelada no fundo") onde a CycleGAN não deu? Motivo: [ADR 0009](../adr/0009-transferencia-neural-como-comparacao.md).

## Previsão (antes de rodar)

Sem previsão registrada.

## Configuração

| Parâmetro | Valor |
|---|---|
| Rede | VGG19 do torchvision (`IMAGENET1K_V1`), max pooling trocado por average pooling |
| Conteúdo / estilo | conv4_2 / Gram de conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 |
| Pesos | estilo 10⁶, conteúdo 1 |
| Otimizador | L-BFGS, 300 iterações, partindo da foto |
| Tamanho | 768 px no maior lado (foto e pintura) |

## Dados

- Foto: Estátua da Liberdade, 1464x2022, céu nublado ([Wikimedia Commons, `Statue_of_Liberty_7.jpg`](https://commons.wikimedia.org/wiki/File:Statue_of_Liberty_7.jpg)), em `data/demo/liberdade.jpg`.
- Estilo: *A Noite Estrelada* (1889), domínio público ([Wikimedia Commons, `Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg`](https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg)), baixada com 1280 px de largura em `data/estilos/noite_estrelada.jpg`.

## Resultado

- Tempo: ~1 min por imagem na RTX 3050 (1m30 na primeira execução, incluindo o download da VGG19, de 548 MB).
- Loss de estilo de 3,03e-05 para 7,88e-08 em 309 avaliações; loss de conteúdo de 0,00 para 0,06.
- Imagens: [`liberdade_noite_estrelada.png`](../../outputs/transferencia/liberdade_noite_estrelada.png) (foto inteira) e [`liberdade_noite_estrelada_ceu.png`](../../outputs/transferencia/liberdade_noite_estrelada_ceu.png) (só o céu).

## O que observei

1. **Bem mais Van Gogh que a CycleGAN na mesma foto.** O céu cinza virou pinceladas curvas, azuis e amarelas, com o ritmo da Noite Estrelada. A CycleGAN, nessa foto, só produziu uma textura granulada ([teste em alta resolução](2026-11-02-nove-blocos.md#teste-em-alta-resolução-0311)).
2. **Na foto inteira, a estátua também é repintada** (fica azul-escura, com pinceladas). Com `--so-ceu`, a estátua mantém a cor e os detalhes originais, e o resultado fica mais próximo da ideia do projeto.
3. As nuvens da foto continuam no lugar: a perda de conteúdo preserva a estrutura e o estilo entra como textura.
4. Há uma borda clara fina em volta da estátua na versão `--so-ceu`, que vem da máscara suavizada.

## Conclusão e próximos passos

A transferência neural entrega a "Noite Estrelada no fundo" que a CycleGAN não entrega, porque copia as estatísticas de textura de um único quadro em vez de aprender o estilo médio de 400. Ela entra na apresentação como comparação, ao lado da CycleGAN (treino 04 ou 06), nas mesmas fotos.
