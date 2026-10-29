# 3. Próximos passos: o que fazer agora

Marque os itens conforme for concluindo.

## Hoje (cerca de 1h): ambiente funcionando

O PyTorch instalado globalmente no computador é a versão só para CPU (não usa a placa de vídeo) e está em conflito com o NumPy 2. Por isso o primeiro passo é um ambiente virtual próprio.

No terminal, dentro da pasta `gan`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

- [ ] Venv criado e ativado
- [ ] PyTorch com CUDA instalado (confirme o comando em https://pytorch.org/get-started/locally/)
- [ ] O último comando imprime `True`

## Depois: entender os dados

O notebook é guiado: você completa os `TODO`s e responde às perguntas. Antes, leia o [guia 05 (KDD)](05-kdd.md).

- [ ] Rodar `jupyter lab`, abrir `notebooks/01_exploracao.ipynb` e completar os TODOs 1 a 10
- [ ] Preencher a tabela de resumo no final do notebook e registrar em [`experimentos/`](../experimentos/)

## Depois: primeiro treino curto, já na GPU

```powershell
python -m src.train_dcgan --epochs 3
```

- [ ] Abrir `outputs/dcgan/samples/` e comparar `epoch_001.png` com `epoch_003.png`
- [ ] Registrar o resultado em [`experimentos/`](../experimentos/)

Ver o borrão virar roupa é o momento em que a GAN começa a fazer sentido.

## Esta semana: confirmar com o professor

O plano original foi sugerido por IA, não é exigência do professor. Confirmar:

- [ ] Formato da entrega (notebook, relatório, apresentação?)
- [ ] Tamanho esperado do texto e critérios de avaliação
- [ ] Individual ou em grupo
- [ ] Se DCGAN + WGAN-GP no Fashion-MNIST é aceitável

Anterior: [Roteiro](02-roteiro-etapas.md) · Próximo: [Como aprender](04-como-aprender.md)
