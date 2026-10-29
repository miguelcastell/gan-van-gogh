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
| `fix` | Correção de bug | `fix(treino): corrige shape do ruído no gerador` |
| `docs` | Só documentação (`docs/`, README, comentários) | `docs(adr): registra escolha da WGAN-GP` |
| `exp` | Resultado de experimento (logs, métricas, figuras, registro em `docs/experimentos/`) | `exp(dcgan): treino de 40 épocas com seed 42` |
| `refactor` | Muda o código sem mudar o comportamento | `refactor(utils): extrai função de salvar grade` |
| `perf` | Melhora de desempenho | `perf(treino): carrega dataset direto na GPU` |
| `style` | Formatação, sem mudar a lógica | `style: padroniza aspas nos scripts` |
| `test` | Testes ou verificações | `test(notebook): adiciona verificação do TODO 7` |
| `chore` | Manutenção: dependências, `.gitignore`, configuração | `chore: adiciona tabulate ao requirements` |

`exp` não faz parte do padrão original. Foi adicionado porque este é um projeto de experimentos e é útil separar "rodei um treino" de "mudei o código".

### Escopos

| Escopo | Parte do projeto |
|---|---|
| `modelos` | `src/models.py` |
| `treino` | `src/train_*.py` |
| `dcgan`, `wgan-gp` | Algo específico de um dos modelos |
| `avaliacao` | `src/evaluate.py`, `src/train_classifier.py`, `src/plots.py` |
| `dados` | Carga e transformação dos dados (`src/utils.py`) |
| `notebook` | `notebooks/` |
| `relatorio` | `report/` |
| `guia`, `adr`, `boas-praticas`, `experimentos` | Subpastas de `docs/` |

Se o commit afeta várias partes, omita o escopo.

## Regras gerais

- **Um assunto por commit.** Se a descrição precisa de "e", provavelmente são dois commits.
- **Autoria somente do autor do projeto.** Sem linhas de coautoria (`Co-Authored-By`) de ferramentas.
- **Não versionar:** o dataset (`data/`), os checkpoints (`outputs/**/checkpoints/`), o ambiente virtual (`.venv/`) e a configuração local (`.claude/`). Tudo já está no `.gitignore`.
- **Versionar:** o código, os notebooks **sem saídas pesadas**, os logs (`logs.csv`), as métricas (`metrics.json`), as grades finais e as figuras do relatório.
- Operações no GitHub (repositório, PRs) são feitas pelo `gh`.
