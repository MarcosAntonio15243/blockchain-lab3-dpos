# blockchain-lab3-dpos

Simulador de Consenso DPoS (Delegated Proof-of-Stake) sob a Ótica Eleitoral.

O motor roda cada cenário por Monte Carlo (várias sementes) e mede a
concentração de poder em três camadas — **stake** (quem detém), **eleito**
(quem foi eleito para o comitê) e **produzido** (quem de fato produziu
blocos) — usando métricas configuráveis (Gini já implementado).

> Nota: o motor (`simulador_dpos.py`) não deve ser reescrito. Cada aluno
> implementa a(s) métrica(s) da sua patologia na seção marcada no arquivo
> (`hhi`, `coef_nakamoto`, `numero_efetivo`, `entropia_shannon`, `palma` ainda
> estão como `TODO` — só `gini` está pronto) e monta o CSV de cenários.

## Requisitos

- Python 3.9+ (testado com 3.14)
- numpy

## Configurando o ambiente com venv

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[dev]"          # projeto + numpy + pytest/black/flake8/mypy
# ou, sem as ferramentas de dev:
# pip install -e .
```

Para sair da venv depois: `deactivate`.

## Executando

### 1. Autoteste + exemplo (sem argumentos)

```bash
python simulador_dpos.py
```

Roda o autoteste do coeficiente de Gini, gera um cenário de exemplo do grupo
G1 (apatia do eleitor) e escreve:

- `cenarios_exemplo.csv` — grade de cenários gerada (3 tamanhos de holders ×
  2 distribuições × 3 níveis de turnout)
- `resultados_exemplo.csv` — resultados agregados (média + IC95) por
  cenário/métrica/camada

### 2. Rodando seu próprio CSV de cenários

```bash
python simulador_dpos.py cenarios.csv resultados.csv
```

- `cenarios.csv`: um cenário por linha, com as colunas de `Config` (veja
  `cenarios_exemplo.csv` como modelo) mais `metrica`, `camada`, `n_runs` e
  `seed_base`. Os campos `metrica` e `camada` aceitam múltiplos valores
  separados por `;` (ex.: `stake;eleito;produzido`).
- `resultados.csv`: saída no formato longo — uma linha por
  `cenário x métrica x camada`, com `media` e `ic95` (intervalo de confiança
  95%) sobre `n_runs` sementes.

## Ferramentas de desenvolvimento (extra `dev`)

```bash
pytest              # se houver testes em tests/
black .             # formatação (line-length 100)
flake8 .            # lint
mypy simulador_dpos.py
```
