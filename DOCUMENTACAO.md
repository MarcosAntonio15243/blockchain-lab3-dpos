# Laboratório 03 — DPoS sob a Ótica Eleitoral

Laboratório sobre **Delegated Proof of Stake (DPoS)** tratado como uma **democracia
representativa censitária**. Cada grupo estuda uma *patologia eleitoral* (apatia,
captura por whales, cartel, compra de voto, etc.), mede seu impacto com métricas de
(des)centralização e submete os dados. A descrição completa, com fórmulas e a divisão
por grupo, está em **`descricao_lab_dpos.pdf`**.

**Sumário**
- [1. Requisitos](#1-requisitos)
- [2. Como funciona o simulador](#2-como-funciona-o-simulador)
- [3. O simulador](#3-o-simulador)
  - [3.1 Como executar](#31-como-executar)
  - [3.2 Formato do `cenarios.csv`](#32-formato-do-cenarioscsv)
  - [3.3 Parâmetros (colunas)](#33-parâmetros-colunas)
  - [3.4 Saída (`resultados.csv`)](#34-saída-resultadoscsv)
- [4. Como implementar as métricas](#4-como-implementar-as-métricas)
- [5. Entrega](#5-entrega)

## 1. Requisitos

- Python 3.9+
- `numpy` → `pip install numpy`

## 2. Como funciona o simulador

Para cada simulação (uma *semente*), o simulador: 
  1) sorteia os **stakes**; 
  2) define como **candidatos** os mais ricos; 
  3) cada holder **vota** com probabilidade
  `turnout`; 
  4) quem vota **aprova** candidatos com prob. proporcional à popularidade, somando o próprio **stake** ao placar; 
  5) os `k` mais votados são **eleitos**; 
  6) os eleitos **produzem blocos** ~ proporcional aos votos. As métricas são calculadas em três **camadas**: `stake` (todos), `eleito` (os `k`) e `produzido` (blocos).

Quando `n_rodadas > 1`, os passos se repetem; entre rodadas os incumbentes podem
ganhar `vantagem_incumbencia` e as recompensas podem ser reinvestidas no stake
via `reinveste_recompensa`.

É um **modelo estocástico de agentes** avaliado por **Monte Carlo** (média ± IC95).
Não é cadeia de Markov. Cartel/suborno trazem elementos de **teoria dos jogos**.

## 3. O simulador

Nas seções abaixo são descritas as funções do simulador

### 3.1 Como executar

A entrada é um `cenarios.csv` (um cenário por linha); a saída é um `resultados.csv`
(formato longo: uma linha por `métrica × camada`):

```bash
python simulador_dpos.py cenarios.csv resultados.csv
```

Exemplo pronto:

```bash
python simulador_dpos.py cenarios_exemplo.csv resultados_exemplo.csv
```

<!-- Você pode utilizar o simulador de notebook, importando e executando como no exemplo abaixo.

```python
from simulador_dpos import gerar_cenarios, salvar_cenarios, rodar_csv

grade = {
    "n_holders":   [200, 500, 1000],          # básico 1
    "distribuicao":["pareto", "lognormal"],   # básico 2
    "turnout":     [0.1, 0.5, 1.0],           # parâmetro da patologia (G1)
}
cen = gerar_cenarios(grade, metricas=["gini"],
                     camadas_alvo=["stake", "eleito", "produzido"],
                     grupo="G1", patologia="apatia_do_eleitor",
                     n_runs=30, seed_base=0)
salvar_cenarios(cen, "cenarios.csv")          # 3*2*3 = 18 cenários
rodar_csv("cenarios.csv", "resultados.csv")   # -> 18*3 camadas = 54 linhas
``` -->

### 3.2 Formato do `cenarios.csv`

Cada linha é **um cenário**. Colunas ausentes assumem o padrão. Duas colunas aceitam
**lista separada por `;`**, e o simulador calcula todas as combinações sobre os
**mesmos sorteios** (comparação justa):

- `metrica` — ex.: `gini;hhi;coef_nakamoto`
- `camada`  — ex.: `stake;eleito;produzido`

Uma linha com `metrica=gini;hhi` e `camada=stake;eleito` gera **4 linhas** de
resultado. Um valor único (`metrica=gini`) também funciona.

### 3.3 Parâmetros (colunas)

**Básicos** — todo grupo varia pelo menos **dois**, com ≥3 valores cada.
**Patologia** — cada grupo varia o seu (≥3 valores).

| Coluna | Tipo / domínio | Faixa sugerida | Efeito ao aumentar | Usa |
|---|---|---|---|---|
| `n_holders` | int ≥ 50 | 200, 500, 1000 | mais eleitores potenciais | básico |
| `distribuicao` | pareto/zipf/lognormal/uniforme | — | muda a forma da riqueza | básico |
| `parametro_dist` | float | pareto: 1.1, 1.5, 2.0 | **menor** = mais concentrado | básico |
| `n_candidatos` | int ≥ `k` | 50, 100, 200 | mais disputa por cadeira | básico |
| `tamanho_comite` | int ≥ 1 | 21, 27, 101 | parlamento maior dispersa poder | básico / **G9** |
| `turnout` | [0,1] | 0.1, 0.5, 1.0 | mais participação | **G1** |
| `n_aprovacoes` | int 1–30 | 30 | cada eleitor apoia mais gente | — |
| `n_blocos` | int | 10000 | granularidade da camada `produzido` | — |
| `frac_proxy` | [0,1] | 0, 0.3, 0.6, 0.9 | mais voto concentrado num proxy | **G5** |
| `tam_cartel` | int 0–`k` | 0, 3, 7, 11 | cartel maior | **G3** |
| `orcamento_suborno` | float ≥ 0 (em stake; lembre que o stake total ≈ 1) | 0, 0.02, 0.05, 0.1 | suborno maior | **G4** |
| `frac_exchange` | [0,1] | 0, 0.2, 0.4, 0.6 | mais stake sob custódia | **G7** |
| `frac_colludida` | [0,1] | 0, 0.2, 0.34, 0.5 | mais delegados em conluio | **G8** |
| `n_rodadas` | int ≥ 1 | 1, 5, 10 | mais eleições encadeadas | **G6/G10** |
| `vantagem_incumbencia` | float ≥ 0 | 0, 0.5, 1.0, 2.0 | incumbente mais protegido | **G6** |
| `reinveste_recompensa` | float ≥ 0 | 0, 0.1, 0.5 | recompensa volta ao stake | **G10** |
| `metrica` | lista `;` | — | — | todos |
| `camada` | lista `;` (`stake`/`eleito`/`produzido`) | — | — | todos |
| `n_runs` | int ≥ 30 | 30 | mais sementes (IC menor) | todos |
| `seed_base` | int | 0 | reprodutibilidade | todos |

> **G6 e G10** usam `n_rodadas > 1`; consultam o **histórico** de rodadas no contexto.

### 3.4 Saída (`resultados.csv`)

Formato **longo**: repete as colunas do cenário e acrescenta o resultado. As duas
primeiras colunas identificam a execução:

- `id_execucao` — igual para todas as linhas de uma
  mesma execução; serve para separar várias execuções/submissões num só arquivo.
- `id_cenario` — índice da linha de entrada (amarra as linhas `métrica × camada` do
  mesmo cenário).

Cada linha traz ainda `metrica`, `camada`, `media`, `ic95`, `n_runs`, `seed_base`.

> ⚠️ **O formato da saída será avaliado automaticamente.** Não altere nem reordene as
> colunas geradas. Você pode acrescentar colunas novas, mas **apenas ao final**. (A
> verificação ignora o *valor* de `id_execucao`, que muda a cada execução.)

## 4. Como implementar as métricas

Cada métrica tem a assinatura **`metrica(ctx) -> float`**. O **Gini já está pronto**
como modelo; implemente a(s) sua(s) na seção `>>> IMPLEMENTE AQUI ... <<<` e registre
no dicionário `METRICAS`. O contexto `ctx` entrega tudo o que você precisa:

| Campo | Para que serve |
|---|---|
| `ctx.shares` | vetor de frações (soma 1) da camada — métricas de **concentração** |
| `ctx.stakes`, `ctx.vota` | turnout por stake/cabeça (G1) |
| `ctx.scores`, `ctx.peso_corte` | custo por cadeira / takeover (G4) |
| `ctx.aprovacoes`, `ctx.idx_diretos` | voto mútuo / Jaccard (G3) |
| `ctx.historico` | rotatividade, persistência, recompensas no tempo (G6/G10) |
| `ctx.cfg` | qualquer parâmetro (ex.: `ctx.cfg.frac_colludida` em G8) |

| Função | Métrica | Fórmula |
|---|---|---|
| `gini` ✅ | Gini | `(2·Σ i·xᵢ)/(n·Σ xᵢ) − (n+1)/n` |
| `hhi` ⬜ | Herfindahl-Hirschman | `Σ sᵢ²` |
| `coef_nakamoto` ⬜ | Coef. de Nakamoto | `min{ m : Σ₁ᵐ s₍ᵢ₎ > limiar }` |
| `numero_efetivo` ⬜ | Nº efetivo (Laakso-Taagepera) | `1 / Σ sᵢ²` |
| `entropia_shannon` ⬜ | Entropia normalizada | `−Σ sᵢ log sᵢ / log N` |
| `palma` ⬜ | Razão de Palma | `share(top 10%) / share(bottom 40%)` |

Só as fórmulas são dadas: pesquise na literatura (e pode usar LLMs para **estudar**) o
que cada métrica significa.

## 5. Entrega

1. **Código** — `simulador_dpos.py` com a(s) métrica(s) implementada(s).
2. **Dados** — `cenarios.csv` (entrada) **e** `resultados.csv` (saída).
3. **Relatório** (3–5 páginas) — explicando **as métricas** e **os resultados
   obtidos**, com gráficos (barras de erro).
4. **Apresentação/Defesa**

> ⚠️ **A escrita do relatório não pode ser feita por ChatGPT ou qualquer LLM** (LLMs
> são permitidos para *estudar* e *implementar* as métricas). **O relatório será
> defendido em sala**, e qualquer integrante poderá ser questionado sobre o código.