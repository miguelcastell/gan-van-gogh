# Experimento: tempo por passo de treino da CycleGAN

- **Data:** 2026-10-29
- **Comando:** testes de velocidade com imagens aleatórias (não o dataset real), na mesma arquitetura de `src/models.py`, e `src/train_cyclegan.py` com 300 fotos falsas (`--epochs 1`, com e sem `--amp`)
- **Pasta de saída:** nenhuma (rodado em pasta temporária)

## Pergunta

Quanto tempo leva um passo de treino da CycleGAN em 128x128 na RTX 3050? Isso define quantas épocas e quantas fotos cabem no prazo.

## Previsão (antes de rodar)

Sem previsão registrada.

## Resultado

- **GPU:** RTX 3050 6 GB Laptop, PyTorch `2.15.0+cu132`, cerca de 5,2 TFLOPS em fp32 (multiplicação de matrizes).
- **Gerador sozinho:** 5,4 ms para converter uma imagem e 16,9 ms para converter e calcular os gradientes.
- **Passo completo (2 geradores e 2 discriminadores, batch 1):**

| Configuração | ms por passo | VRAM (pico) |
|---|---|---|
| `cudnn.benchmark` desligado | 741 | 712 MB |
| `cudnn.benchmark` ligado | 125 a 138 | 2,6 GB |
| `cudnn.benchmark` ligado + `--amp` (bf16) | 85 a 124 | 0,5 GB |

A primeira medição com o `cudnn.benchmark` ligado deu 328 ms. As seguintes ficaram entre 125 e 138 ms, então ela provavelmente foi distorcida pelo aquecimento da GPU.

## O que observei

- O passo é lento em relação ao gerador sozinho porque a CycleGAN roda um gerador **6 vezes** por passo (ida, volta e *identity*, nos dois sentidos), além dos dois discriminadores.
- O `cudnn.benchmark` faz o cuDNN testar os algoritmos de convolução e ficar com o mais rápido. Como as imagens têm sempre o mesmo tamanho, isso vale a pena e passou a ficar sempre ligado no `src/train_cyclegan.py`.
- O `--amp` ajuda pouco no passo inteiro, mas usa muito menos memória da GPU. Ficou como opção.

## Conclusão e próximos passos

Com cerca de 130 ms por passo:
- uma época com todas as ~6 mil fotos leva cerca de 13 minutos, e 40 épocas, quase 9 horas;
- com `--photos 2000`, uma época leva cerca de 4 minutos, e 40 épocas, cerca de 3 horas.

Próximo passo: medir de novo com o dataset real (`--epochs 1 --photos 300`) e decidir épocas e fotos ([ADR 0002](../adr/0002-dataset-e-modelos.md), "Em aberto").
