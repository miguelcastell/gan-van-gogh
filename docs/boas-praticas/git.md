# Git e commits

## Padrão de mensagem: Conventional Commits

Toda mensagem de commit segue o formato do [Conventional Commits](https://www.conventionalcommits.org/pt-br/):

```
tipo(escopo): descrição curta no imperativo

Corpo opcional explicando o porquê da mudança.
```

- **tipo:** o que o commit faz (tabela abaixo). Obrigatório.
- **escopo:** a parte do projeto afetada (tabela abaixo). Opcional, mas recomendado.
- **descrição:** em português, minúscula, sem ponto final, no imperativo ("adiciona", "corrige"), com até ~70 caracteres.

**Por quê:** o histórico fica legível de relance, e dá para saber o que cada commit mudou sem abrir o código.

### Tipos

| Tipo | Quando usar | Exemplo |
|---|---|---|
| `feat` | Funcionalidade nova (modelo, métrica, script, análise) | `feat(avaliacao): adiciona cálculo de FID` |
| `fix` | Correção de bug | `fix(treino): corrige shape da máscara do céu` |
| `docs` | Só documentação (`docs/`, README, comentários) | `docs(adr): registra escolha do segmentador de céu` |
| `exp` | Resultado de experimento (logs, métricas, figuras, registro em `docs/experimentos/`) | `exp(cyclegan): treino de 40 épocas com seed 42` |
| `refactor` | Muda o código sem mudar o comportamento | `refactor(utils): extrai função de salvar grade` |
| `perf` | Melhora de desempenho | `perf(dados): carrega as imagens uma vez em memória` |
| `style` | Formatação, sem mudar a lógica | `style: padroniza aspas nos scripts` |
| `test` | Testes ou verificações | `test(notebook): adiciona verificação do TODO 7` |
| `chore` | Manutenção: dependências, `.gitignore`, configuração | `chore: adiciona tabulate ao requirements` |

`exp` não faz parte do padrão original. Foi adicionado porque este é um projeto de experimentos e é útil separar "rodei um treino" de "mudei o código".

### Escopos

| Escopo | Parte do projeto |
|---|---|
| `modelos` | `src/models.py` |
| `treino` | `src/train_*.py` |
| `cyclegan`, `sem-cycle` | Algo específico de um dos experimentos |
| `ceu` | Máscara do céu e composição (`src/sky.py`, `src/stylize.py`) |
| `avaliacao` | `src/evaluate.py`, `src/plots.py` |
| `dados` | Carga e transformação dos dados (`src/utils.py`) |
| `notebook` | `notebooks/` |
| `relatorio` | `report/` |
| `guia`, `adr`, `boas-praticas`, `experimentos` | Subpastas de `docs/` |

Se o commit afeta várias partes, omita o escopo.

## Regras gerais

- **Um assunto por commit.** Se a descrição precisa de "e", provavelmente são dois commits.
- **Autoria somente do autor do projeto.** Sem linhas de coautoria (`Co-Authored-By`) de ferramentas.
- **Não versionar:** o dataset (`data/`), os checkpoints (`outputs/**/checkpoints/`), o ambiente virtual (`.venv/`) e a configuração local (`.claude/`). Tudo já está no `.gitignore`.
- **Versionar:** o código, os notebooks **sem saídas**, os logs (`logs.csv`), as métricas (`metrics.json`), as grades das épocas múltiplas de 5 e as figuras do relatório. As outras grades e o `evolucao.gif` ficam só no computador local.
  **Por quê:** o git guarda tudo para sempre. Com todas as grades e o GIF, cada treino somaria ~45 MB ao histórico; só com as de 5 em 5, ~7 MB.
- **Saídas dos notebooks nunca vão para o git.** O [nbstripout](https://github.com/kynan/nbstripout) apaga as saídas automaticamente no `git add`, só na versão versionada: no seu computador o notebook continua com tudo. As respostas ✍️ ficam, porque são células de texto. Está configurado pelo `.gitattributes`, mas o filtro em si fica no `.git/config`, que não é versionado: em um clone novo, rode `nbstripout --install`.
  **Por quê:** as saídas têm imagens e gráficos (um notebook executado passa de 8 MB), mudam a cada execução e poluiriam o histórico e os diffs. As figuras que importam vão para `report/figuras/`.
- Operações no GitHub (repositório, PRs) são feitas pelo `gh`.
