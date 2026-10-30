# 2. Roteiro: as etapas do projeto

Os prazos vêm do plano em [`projeto-final-gan.md`](../../projeto-final-gan.md).

## Etapa 1: Ambiente e dados (até 04/11)

- **O que você aprende:** como uma imagem colorida vira números. Cada pixel tem 3 valores (vermelho, verde e azul), e cada um vira um número entre -1 e 1.
- **Arquivo:** `notebooks/01_exploracao.ipynb`
- **O que observar:** como as pinturas e as fotos diferem em cores e textura, e o desequilíbrio entre os dois conjuntos.

## Etapa 2: CycleGAN completa (até 11/11)

- **O que você aprende:** o gerador que recebe uma imagem, o discriminador que julga pedaços da imagem (PatchGAN) e as três losses: adversarial, *cycle* e *identity*.
- **Arquivos:** `notebooks/02_cyclegan.ipynb` (você monta as peças e faz um treino curto), depois `src/train_cyclegan.py` no terminal (treino longo)
- **O que observar:**
  - A grade de fotos convertidas melhorando época a época (o GIF em `outputs/cyclegan/evolucao.gif`).
  - A *cycle loss* caindo: o modelo está aprendendo a ir e voltar sem perder o conteúdo.
  - As losses dos discriminadores: se uma delas vai a zero, aquele D venceu o jogo.

## Etapa 3: Comparação e máscara do céu (até 18/11)

- **O que você aprende:** para que serve cada parte da loss, e como juntar dois modelos num sistema.
- **Arquivos:** `src/train_cyclegan.py --sem-cycle`, `src/sky.py`, `src/stylize.py`
- **O que observar:**
  - Sem a *cycle loss*, o conteúdo da foto se mantém ou some?
  - A máscara pega o céu certo? A emenda entre o céu pintado e o resto aparece?

## Etapa 4: Avaliação (até 25/11)

- **O que você aprende:** como medir algo subjetivo como "isso parece um Van Gogh?".
- **Arquivos:** `src/evaluate.py`, `src/plots.py`
- **O que observar:**
  - O **FID** e o **KID**: quanto menor, mais as saídas se parecem com as pinturas reais.
  - O erro de ida e volta: quanto do conteúdo foi preservado.
  - Pares "saída x pintura de treino mais parecida" (memorização).

## Etapa 5: Escrita (até 02/12)

- **O que você aprende:** a interpretar os resultados e conectá-los com a ética.
- **Arquivo:** `report/analise_etica.md`
- O texto deve se basear nas suas figuras, não em teoria genérica.

Anterior: [Visão geral](01-visao-geral.md) · Próximo: [Próximos passos](03-proximos-passos.md)
