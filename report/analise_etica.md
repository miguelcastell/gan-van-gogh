# Análise ética e crítica

> Regra: cada ponto cita uma evidência do próprio projeto (figura, métrica ou exemplo).
> Modelo de referência: treino 04 (`cyclegan_256`, 256x256, época 40), salvo quando indicado.

## 1. Imitação de estilo e autoria

**No projeto.** O estilo vem de 400 pinturas de Van Gogh (`trainA` do `vangogh2photo`, que o artigo da CycleGAN montou a partir do WikiArt). Van Gogh morreu em 1890: pela Lei 9.610/1998 (art. 41), os direitos patrimoniais duram 70 anos após a morte do autor, então a obra está em domínio público. O mesmo vale para a *Noite Estrelada* usada na transferência neural ([ADR 0009](../docs/adr/0009-transferencia-neural-como-comparacao.md)).

**Se fosse um artista vivo.** A técnica é a mesma: bastam algumas centenas de obras para um modelo produzir imagens "no estilo de" qualquer pessoa. A transferência neural precisa de uma única obra. O artista não é consultado, não é pago e passa a competir com cópias do próprio estilo. Casos reais:
- Greg Rutkowski, ilustrador polonês, teve o nome usado em centenas de milhares de *prompts* do Stable Diffusion em 2022 e se manifestou contra (MIT Technology Review, set. 2022).
- *Andersen v. Stability AI* (EUA, 2023): processo coletivo de artistas contra empresas de geração de imagens pelo uso das obras deles no treino.
- *Getty Images v. Stability AI* (Reino Unido e EUA, 2023): o banco de imagens processou pelo uso de fotos dele no treino. Saídas do modelo chegavam a reproduzir a marca d'água da Getty.

**Quem é o autor do resultado?** A foto é do fotógrafo, o estilo é de Van Gogh, e o modelo foi treinado por nós. Na nossa leitura, o resultado é uma obra derivada da foto: o fotógrafo continua tendo direitos sobre ela.

## 2. As fotos do dataset também têm dono

**Evidência:** a figura [`cyclegan_256_vizinhos.png`](figuras/cyclegan_256_vizinhos.png) mostra saídas com marcas como **"© Juan DeLeon Studio (DO NOT COPY)"**, e a [`cyclegan_256_piores.png`](figuras/cyclegan_256_piores.png) mostra "© Karlton Huber". As 6.287 fotos de `trainB` vieram do Flickr. Pelo menos parte delas tem aviso de direitos autorais e foi usada sem consentimento explícito dos fotógrafos. O gerador ainda **reproduz a marca d'água pintada**, que é o mesmo fenômeno do caso da Getty.

Para um trabalho acadêmico, que usa um dataset público e conhecido, sem fins comerciais, o risco é baixo. Para um produto, seria um problema de licenciamento.

## 3. Manipulação de fotos e desinformação

**No projeto.** Trocar o céu é editar a realidade. Num estilo de pintura, a edição é óbvia: ninguém confunde o resultado com uma foto. Mas o pipeline (segmentar uma região e substituí-la por conteúdo gerado) é exatamente o de uma edição realista: com outro gerador, daria para trocar um céu limpo por fumaça, ou um céu de dia por um de tempestade. A borda suave da máscara ([`sky.py`](../src/sky.py)) existe justamente para esconder a emenda.

**Evidência de que a emenda aparece:** no teste com a Estátua da Liberdade ([experimento](../docs/experimentos/2026-11-02-nove-blocos.md#teste-em-alta-resolução-0311)), com `--size 256` fica um halo visível em volta da estátua. Hoje, a edição ainda é detectável a olho nu.

**Casos reais:**
- Foto falsa do Papa Francisco com um casaco branco acolchoado, gerada no Midjourney (mar. 2023), circulou como real.
- Imagem falsa de uma explosão perto do Pentágono (mai. 2023) se espalhou em contas verificadas e coincidiu com uma queda breve na bolsa americana.

**Se fosse aplicado a rostos ou a fotos jornalísticas:** com rostos, entra a LGPD (seção 5). Em foto jornalística, qualquer alteração, mesmo "artística", muda o valor de prova da imagem.

## 4. Viés

**Evidência:** os piores casos ([`cyclegan_256_piores.png`](figuras/cyclegan_256_piores.png), colunas foto | G(foto) | F(G(foto))):
- **Noite vira dia:** o céu estrelado da 2ª linha virou laranja e azul-claro. Das 400 pinturas, poucas são noturnas (a *Noite Estrelada*, o *Terraço do Café à Noite*), então o modelo "acha" que pintura de Van Gogh é diurna e amarela.
- **Close-ups e texturas uniformes** (a flor, o céu cheio de nuvenzinhas, a 1ª e a 3ª linhas): fora do que o dataset tem (paisagens), o gerador inventa manchas.
- **Céu cinza e liso** (estátua; céu da casinha na galeria): textura granulada em vez de pincelada.

**Viés de dados:**
- **Fotos:** são paisagens do Flickr, escolhidas por *tags*. Quase não há pessoas, cidades à noite ou interiores, e as fotos refletem os lugares e o gosto de quem publica no Flickr.
- **Pinturas:** as 400 são uma amostra das ~860 pinturas a óleo de Van Gogh. A coleção mistura fases bem diferentes (os tons escuros dos Países Baixos e as cores fortes de Arles), e o modelo aprende uma média delas.
- **Componentes pré-treinados:** o segmentador (SegFormer no ADE20K) e a métrica (Inception no ImageNet) trazem os vieses das próprias bases.

**Seleção do treino 06:** escolher só fotos de céu "interessante" ([ADR 0008](../docs/adr/0008-treinar-com-fotos-de-ceu.md)) é um viés introduzido de propósito. O modelo fica melhor em pôr do sol e pior em céus comuns.

## 5. Memorização e privacidade

**Evidência:** a distância média de cada saída até a pintura mais parecida é **198,7**, e a distância entre duas pinturas reais diferentes é **203,0** ([`tabela_comparativa.md`](tabela_comparativa.md)). A menor distância de uma saída foi 131,5, longe de 0. Na [`cyclegan_256_vizinhos.png`](figuras/cyclegan_256_vizinhos.png), as 8 saídas mais próximas de uma pintura (a *Cadeira de Gauguin*) são fotos noturnas que apenas compartilham a paleta escura. Nenhuma copia a cadeira. **Não há sinal de memorização.** Isso faz sentido para a CycleGAN: o gerador recebe uma foto, não ruído, e a *cycle loss* obriga a saída a conter a foto.

**Por que isso importa:** modelos generativos podem memorizar e reproduzir dados de treino. Carlini et al. (2023) extraíram imagens quase idênticas às de treino de modelos de difusão. Se a base tiver obras protegidas, a cópia viola direitos. Se tiver fotos de pessoas, vaza dados pessoais.

**LGPD (Lei 13.709/2018):** a foto de uma pessoa identificável é dado pessoal (art. 5º, I), e dados biométricos, como o rosto, são dados sensíveis (art. 5º, II), com regras mais rígidas de tratamento (art. 11). Este projeto usa paisagens. Treinar com fotos de pessoas, ou aplicar o modelo a fotos de terceiros, exigiria base legal, normalmente o consentimento.

## 6. Limites das métricas

- **FID com poucas imagens:** o FID é enviesado para amostras pequenas (Chong e Forsyth, 2020), e temos só 400 pinturas como referência. Por isso também medimos o KID, que não tem esse viés. A diferença de 1,2 ponto de FID entre os treinos 04 e 05 está dentro da oscilação de ~4 pontos entre checkpoints vizinhos, ou seja, é ruído.
- **O FID mede o estilo, não a preservação do conteúdo.** Um gerador que ignora a foto e produz "pinturas genéricas" poderia ter um FID razoável. É por isso que medimos também o erro de ida e volta (L1) e a diversidade. O treino sem *cycle loss* mostrou o caso extremo: colapso para quase a mesma imagem (diversidade 56,9 contra 137,7 das pinturas, erro de ida e volta 0,494 contra 0,130).
- **O FID usa a Inception (ImageNet), uma rede de fotos.** Ela não foi feita para julgar pinceladas, e o FID reage a classes do ImageNet (Kynkäänniemi et al., 2023). A métrica concordou pouco com a impressão visual: o treino 04 foi o mais bonito, mas o de melhor FID foi o treino 03 (117,4).
- **Nenhuma métrica mede "parece Van Gogh" para uma pessoa.** A avaliação visual (galeria, comparação lado a lado) foi decisiva nas escolhas do projeto.

## 7. Medidas de mitigação

- **Marcar o conteúdo sintético:** metadados de proveniência, como o padrão C2PA (*Content Credentials*), ou uma marca visível ao publicar as imagens.
- **Documentar a base:** uma ficha no estilo *Datasheets for Datasets* (Gebru et al., 2021), dizendo de onde vêm as fotos e as pinturas e quais licenças elas têm. A [ficha dos dados](../docs/referencia/dados.md) do projeto é um começo.
- **Consentimento:** só treinar com obras de artistas vivos com autorização. Respeitar *opt-out* e avisos como "DO NOT COPY".
- **Filtrar marcas d'água** antes do treino, para o gerador não aprender a reproduzi-las.
- **Medir memorização** sempre (vizinho mais próximo), como foi feito aqui.

## 8. Análise crítica

**O que funcionou:** a *cycle loss* é essencial (sem ela, colapso); o recorte aleatório melhorou o FID (125,3 → 117,4); o treino em 256 deu as primeiras pinceladas visíveis.

**O que não funcionou:** mais capacidade (9 blocos) não ajudou com 400 pinturas. A CycleGAN não cria a "Noite Estrelada no fundo": ela troca cor e textura, mas não inventa estrutura (redemoinhos, estrelas) onde a foto não tem, porque as perdas de ciclo e de identidade a obrigam a preservar a foto. Em céus cinza e lisos, o resultado é uma textura granulada.

**Limite fundamental:** a CycleGAN aprende o estilo médio de um pintor; a transferência neural copia um quadro. Na mesma foto da estátua, a transferência neural deu as pinceladas curvas da *Noite Estrelada* em ~1 min, sem treino nenhum ([experimento](../docs/experimentos/2026-11-03-transferencia-neural.md)). Para o objetivo estético, ela venceu. Mas a GAN generaliza: depois de treinada, converte qualquer foto em milissegundos e captura um estilo, não uma obra.

**O que faria diferente com mais tempo e GPU:**
- treinar por 200 épocas, como no artigo (aqui foram 30 a 40, pela restrição de ~1 noite por treino numa RTX 3050);
- curar as pinturas por fase (por exemplo, só Arles e Saint-Rémy);
- treinar em recortes só do céu;
- testar modelos mais recentes de tradução de imagem.

## Referências

- Zhu, J.-Y. et al. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV.
- Gatys, L. A., Ecker, A. S. e Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks*. CVPR.
- Heusel, M. et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID). NeurIPS.
- Bińkowski, M. et al. (2018). *Demystifying MMD GANs* (KID). ICLR.
- Chong, M. J. e Forsyth, D. (2020). *Effectively Unbiased FID and Inception Score and Where to Find Them*. CVPR.
- Kynkäänniemi, T. et al. (2023). *The Role of ImageNet Classes in Fréchet Inception Distance*. ICLR.
- Carlini, N. et al. (2023). *Extracting Training Data from Diffusion Models*. USENIX Security.
- Gebru, T. et al. (2021). *Datasheets for Datasets*. Communications of the ACM.
- Brasil. Lei nº 9.610/1998 (Lei de Direitos Autorais), art. 41.
- Brasil. Lei nº 13.709/2018 (LGPD), arts. 5º e 11.
- C2PA — Coalition for Content Provenance and Authenticity. https://c2pa.org
- Heikkilä, M. (2022). *This artist is dominating AI-generated art. And he's not happy about it.* MIT Technology Review.
