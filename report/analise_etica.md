# Análise ética e crítica

> Regra: cada ponto deve citar uma evidência do próprio projeto (figura, métrica ou exemplo).

## 1. Imitação de estilo e autoria
- Van Gogh é domínio público. O que muda se a mesma técnica for treinada com obras de um artista vivo?
- Quem é o autor da foto com céu de Van Gogh: o fotógrafo, o pintor, quem treinou o modelo?
- Casos reais: (referências)

## 2. Manipulação de fotos e desinformação
- Trocar o céu é editar a realidade. Evidência: `figuras/*_ceu.png`. Dá para perceber que a foto foi alterada?
- O que mudaria se a técnica fosse aplicada a fotos jornalísticas ou a rostos?
- Casos reais: (referências)

## 3. Viés
- Evidência: exemplos ruins. Que tipos de foto o modelo trata mal (noite, interiores, pessoas)?
- As fotos do dataset são paisagens do Flickr, e o estilo vem de um único pintor. Que parte da obra de Van Gogh as ~400 pinturas representam?

## 4. Memorização
- Evidência: `figuras/*_vizinhos.png`. As saídas copiam pedaços de pinturas específicas?
- Implicações para obras protegidas e para fotos de pessoas (LGPD).

## 5. Limites das métricas
- O FID com poucas centenas de pinturas é confiável? Comparar com o KID.
- O FID mede o estilo, não a preservação do conteúdo: a versão sem *cycle loss* pode ter FID melhor e resultado pior.

## 6. Medidas de mitigação
- Marcação de conteúdo sintético, documentação da base, consentimento de artistas.

## 7. Análise crítica
- Limitações das métricas, o que deu errado, o que faria diferente.

## Referências
