# Análise ética e crítica

> Regra: cada ponto deve citar uma evidência do próprio projeto (figura, métrica ou exemplo).

## 1. Deepfakes e desinformação
- O que mudaria se a mesma técnica fosse aplicada a rostos?
- Casos reais: (referências)

## 2. Viés
- Evidência: `figuras/*_classes.png`. Quais classes o gerador aprendeu bem/mal?
- Que escolhas e exclusões a base Fashion-MNIST reflete?

## 3. Privacidade e memorização
- Evidência: `figuras/*_vizinhos.png`, `figuras/*_memorizacao.png` (gerado→treino vs. teste→treino).
- Implicações para dados pessoais e LGPD.

## 4. Dados sintéticos usados como reais
- Riscos de treinar outros modelos com dados gerados; colapso ao longo de gerações.

## 5. Limites das métricas
- O que o FID não captura? Por que boas métricas não garantem ausência de viés.

## 6. Medidas de mitigação
- Marcação de conteúdo sintético, documentação da base, auditoria de amostras.

## 7. Análise crítica
- Limitações das métricas, o que deu errado, o que faria diferente.

## Referências
