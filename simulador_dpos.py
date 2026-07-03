"""
Simulador de Consenso DPoS sob a Ótica Eleitoral
================================================

NOTA: CODIGO GERADO POR IA

Motor COMUM do laboratorio. Voce (aluno) NAO reescreve o motor: voce
(1) implementa a(s) metrica(s) da sua patologia na secao marcada mais abaixo
(o Gini ja vem pronto como modelo) e (2) monta um CSV de CENARIOS com as
combinacoes de parametros que quer testar. O simulador roda cada cenario por
Monte Carlo e gera um CSV de RESULTADOS no formato LONGO (uma linha por
metrica x camada), com um id de execucao e um id de cenario.

Uso:
    python simulador_dpos.py cenarios.csv resultados.csv
    python simulador_dpos.py            # autoteste do Gini + exemplo

------------------------------------------------------------------------------
COMO O MOTOR FUNCIONA (passo a passo, para cada semente)
------------------------------------------------------------------------------
  1. GERACAO DE STAKES: sorteia o stake de `n_holders` detentores segundo
     `distribuicao` (pareto/zipf/lognormal/uniforme); `parametro_dist` controla
     a desigualdade.
  2. CANDIDATURA: os `n_candidatos` mais ricos sao candidatos; a popularidade de
     cada um e proporcional ao seu stake.
  3. PARTICIPACAO: cada holder vota com probabilidade `turnout`.
  4. VOTACAO POR APROVACAO PONDERADA: cada votante aprova candidatos com prob.
     proporcional a popularidade (~`n_aprovacoes` aprovacoes); cada aprovacao
     soma o STAKE do eleitor ao placar do candidato.
  5. ELEICAO: os `tamanho_comite` (k) candidatos com maior placar sao eleitos.
  6. PRODUCAO: os eleitos produzem `n_blocos` blocos ~ proporcional aos votos.
  7. METRICAS em tres camadas: stake / eleito / produzido.

Quando `n_rodadas` > 1, os passos 1-6 se repetem em rodadas; entre elas, os
incumbentes podem ganhar `vantagem_incumbencia` (G6) e as recompensas podem ser
reinvestidas no stake via `reinveste_recompensa` (G10). O historico de rodadas
fica disponivel no contexto.

Natureza: MODELO ESTOCASTICO de agentes, avaliado por MONTE CARLO (media +/-
IC95 sobre varias sementes). NAO e cadeia de Markov (cada execucao e
independente). Cartel/suborno trazem TEORIA DOS JOGOS (cf. arXiv:2310.18596).

------------------------------------------------------------------------------
OBJETO DE CONTEXTO (o que sua metrica recebe)
------------------------------------------------------------------------------
Toda metrica tem a assinatura `metrica(ctx) -> float`. O `ctx` carrega:
  ctx.shares      : vetor de frações (soma 1) da camada pedida  [concentracao]
  ctx.camada      : 'stake' | 'eleito' | 'produzido'
  ctx.stakes      : stake de todos os holders (normalizado)
  ctx.scores      : placar de cada candidato (alinhado a ctx.pool)
  ctx.eleitos     : indices (no pool) dos k eleitos
  ctx.pool        : indices (de holders) dos candidatos
  ctx.peso_corte  : placar do (k+1)-esimo candidato (o "compravel")
  ctx.vota        : mascara booleana de quem votou (turnout)
  ctx.aprovacoes  : matriz (votantes_diretos x candidatos) de aprovacoes
  ctx.idx_diretos : indices dos votantes diretos (linhas de ctx.aprovacoes)
  ctx.blocos      : fracao de blocos por eleito
  ctx.historico   : lista de rodadas (cada uma com scores/eleitos/blocos/stakes)
  ctx.cfg         : a Config do cenario (le qualquer parametro, ex.: frac_colludida)

Requisitos: Python 3.9+ e numpy  (pip install numpy). Veja o README.md.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from itertools import product
from typing import Callable, Sequence

import numpy as np


# ===========================================================================
# 1) ENTRADAS (uma instancia de Config = um cenario)
# ===========================================================================
@dataclass
class Config:
    # metadados
    grupo: str = "demo"
    patologia: str = "exemplo"
    # basicos (todo grupo varia >= 2 destes, com >= 3 valores cada)
    n_holders: int = 1000
    distribuicao: str = "pareto"
    parametro_dist: float = 1.16
    n_candidatos: int = 100
    tamanho_comite: int = 21
    # patologia (cada grupo varia o seu)
    turnout: float = 1.0              # G1 apatia
    n_aprovacoes: int = 30
    n_blocos: int = 10_000
    frac_proxy: float = 0.0           # G5 proxy
    tam_cartel: int = 0               # G3 cartel
    orcamento_suborno: float = 0.0    # G4 suborno
    frac_exchange: float = 0.0        # G7 custodia
    frac_colludida: float = 0.0       # G8 censura (usado pela metrica)
    # multi-rodada (G6, G10)
    n_rodadas: int = 1
    vantagem_incumbencia: float = 0.0  # G6 entrincheiramento
    reinveste_recompensa: float = 0.0  # G10 recompensas compostas


PARAMS = [f.name for f in fields(Config)]
# Esquema canonico (NAO altere/reordene; novos campos so AO FINAL):
CAMPOS_ENTRADA = PARAMS + ["metrica", "camada", "n_runs", "seed_base"]
CAMPOS_SAIDA = (["id_execucao", "id_cenario"] + PARAMS
                + ["metrica", "camada", "media", "ic95", "n_runs", "seed_base"])

_TIPO = {
    "grupo": str, "patologia": str, "distribuicao": str,
    "n_holders": int, "n_candidatos": int, "tamanho_comite": int,
    "n_aprovacoes": int, "n_blocos": int, "tam_cartel": int, "n_rodadas": int,
    "parametro_dist": float, "turnout": float, "frac_proxy": float,
    "orcamento_suborno": float, "frac_exchange": float, "frac_colludida": float,
    "vantagem_incumbencia": float, "reinveste_recompensa": float,
}


# ===========================================================================
# 2) MOTOR ELEITORAL (nao precisa alterar)
# ===========================================================================
def gerar_stakes(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.distribuicao == "pareto":
        s = rng.pareto(cfg.parametro_dist, cfg.n_holders) + 1.0
    elif cfg.distribuicao == "zipf":
        s = rng.zipf(max(cfg.parametro_dist, 1.01), cfg.n_holders).astype(float)
    elif cfg.distribuicao == "lognormal":
        s = rng.lognormal(mean=0.0, sigma=cfg.parametro_dist, size=cfg.n_holders)
    elif cfg.distribuicao == "uniforme":
        s = rng.random(cfg.n_holders) + 1e-9
    else:
        raise ValueError(f"distribuicao desconhecida: {cfg.distribuicao}")
    return s / s.sum()


def realizar_eleicao(stakes, cfg, rng, incumbentes=None) -> dict:
    n = len(stakes)
    n_cand = min(cfg.n_candidatos, n)
    pool = np.argsort(stakes)[::-1][:n_cand]
    stake_pool = stakes[pool]

    pop = stake_pool.copy()
    # G6 incumbencia: incumbentes (holders) ganham vantagem na reeleicao
    if cfg.vantagem_incumbencia > 0 and incumbentes:
        mask = np.isin(pool, np.fromiter(incumbentes, dtype=int))
        pop[mask] *= (1.0 + cfg.vantagem_incumbencia)
    # G4 suborno: injeta orcamento no candidato logo abaixo do corte
    if cfg.orcamento_suborno > 0 and n_cand > cfg.tamanho_comite:
        pop[cfg.tamanho_comite] += cfg.orcamento_suborno
    pop = pop / pop.sum()

    n_aprov = min(cfg.n_aprovacoes, n_cand)
    p_aprova = np.clip(n_aprov * pop, 0.0, 1.0)

    vota = rng.random(n) < cfg.turnout
    idx = np.where(vota)[0]
    scores = np.zeros(n_cand)

    # G5 proxy: uma fracao dos votantes delega a UM proxy (vota em bloco)
    if cfg.frac_proxy > 0 and len(idx) > 0:
        n_proxy = int(cfg.frac_proxy * len(idx))
        deleg, diretos = idx[:n_proxy], idx[n_proxy:]
        if n_proxy > 0:
            cesta = (rng.random(n_cand) < p_aprova).astype(float)
            scores += stakes[deleg].sum() * cesta
    else:
        diretos = idx

    if len(diretos) > 0:
        aprov = rng.random((len(diretos), n_cand)) < p_aprova
        scores += stakes[diretos] @ aprov
    else:
        aprov = np.zeros((0, n_cand), dtype=bool)

    # G3 cartel: os tam_cartel mais ricos votam uns nos outros
    if cfg.tam_cartel > 0:
        c = min(cfg.tam_cartel, n_cand)
        scores[:c] += stake_pool[:c].sum()

    ordem = np.argsort(scores)[::-1]
    eleitos = ordem[:cfg.tamanho_comite]
    peso_corte = float(scores[ordem[cfg.tamanho_comite]]) if n_cand > cfg.tamanho_comite else 0.0

    return {"scores": scores, "eleitos": eleitos, "pool": pool,
            "peso_corte": peso_corte, "vota": vota,
            "aprovacoes": aprov, "idx_diretos": diretos}


def produzir_blocos(scores, eleitos, cfg, rng) -> np.ndarray:
    p = scores[eleitos].astype(float)
    if p.sum() <= 0:
        p = np.ones_like(p)
    p = p / p.sum()
    counts = rng.multinomial(cfg.n_blocos, p).astype(float)
    return counts / counts.sum() if counts.sum() > 0 else p


def simular(cfg: Config, rng: np.random.Generator) -> dict:
    """Roda `n_rodadas` eleicoes, encadeando estado (incumbencia/recompensa).
    Devolve o 'bundle' da ultima rodada + o historico de todas as rodadas."""
    stakes = gerar_stakes(cfg, rng)
    historico, incumbentes, ultima = [], None, None
    for _ in range(max(1, cfg.n_rodadas)):
        el = realizar_eleicao(stakes, cfg, rng, incumbentes)
        blocos = produzir_blocos(el["scores"], el["eleitos"], cfg, rng)
        rod = dict(el)
        rod["blocos"] = blocos
        rod["stakes"] = stakes.copy()
        historico.append(rod)
        incumbentes = set(el["pool"][el["eleitos"]].tolist())
        # G10 recompensas compostas: reinveste no stake dos eleitos
        if cfg.reinveste_recompensa > 0:
            stakes = stakes.copy()
            stakes[el["pool"][el["eleitos"]]] += cfg.reinveste_recompensa * blocos
            stakes = stakes / stakes.sum()
        ultima = rod
    bundle = dict(ultima)
    bundle["historico"] = historico
    return bundle


def camadas(b: dict) -> dict:
    se = b["scores"][b["eleitos"]].astype(float)
    eleito = se / se.sum() if se.sum() > 0 else np.ones(len(b["eleitos"])) / len(b["eleitos"])
    return {"stake": b["stakes"] / b["stakes"].sum(), "eleito": eleito, "produzido": b["blocos"]}


def agregar_exchange(shares, frac: float) -> np.ndarray:
    """G7: agrega as maiores entidades que somam ~`frac` do total numa unica."""
    if frac <= 0:
        return shares
    s = np.sort(np.asarray(shares, dtype=float))[::-1]
    total, acc, idx = s.sum(), 0.0, 0
    while idx < len(s) and acc < frac * total:
        acc += s[idx]
        idx += 1
    return s if idx <= 1 else np.concatenate(([s[:idx].sum()], s[idx:]))

# ===========================================================================
# 3) OBJETO DE CONTEXTO
# ===========================================================================
@dataclass
class Contexto:
    cfg: Config
    camada: str
    shares: np.ndarray
    stakes: np.ndarray
    scores: np.ndarray
    eleitos: np.ndarray
    pool: np.ndarray
    peso_corte: float
    vota: np.ndarray
    aprovacoes: np.ndarray
    idx_diretos: np.ndarray
    blocos: np.ndarray
    historico: list


# ===========================================================================
# 4) BIBLIOTECA DE METRICAS
# ===========================================================================
# Toda metrica recebe `ctx` (Contexto) e devolve um float. O Gini ja vem pronto.
# IMPORTANTE: aqui estao apenas as FORMULAS/assinaturas. Pesquise na literatura
# (e pode usar LLMs para ESTUDAR) o que cada metrica significa.

def _gini(shares) -> float:
    x = np.sort(np.asarray(shares, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2.0 * np.sum(i * x)) / (n * total) - (n + 1) / n)


def gini(ctx: Contexto) -> float:
    """Coeficiente de Gini (0 = igualdade; ->1 = concentracao maxima)."""
    return _gini(ctx.shares)


# ---------------------------------------------------------------------------
# >>> IMPLEMENTE AQUI AS DEMAIS METRICAS DA SUA PATOLOGIA <<<
# Apague o `raise` ao implementar. Veja gini() como modelo. Use os campos do
# contexto. Exemplos de uso (NAO implementados de proposito):
#
#   # HHI (concentracao, usa shares):
#
#   # Coef. de Nakamoto (concentracao, usa shares):
#
#   # turnout por stake (G1, usa ctx.vota e ctx.stakes):
#
#   # custo por cadeira (G4, usa ctx.scores / ctx.peso_corte):
#
#   # censura (G8, usa ctx.cfg.frac_colludida sobre a camada 'eleito'):
#   #   peso colludido dos eleitos; compara com 1/3
#
#   # rotatividade (G6, usa ctx.historico, multi-rodada):
#   #   compara conjuntos ctx.historico[t]['eleitos'] entre rodadas
#
#   # recompensa acumulada (G10, multi-rodada):
#   #   soma ctx.historico[t]['blocos'] por holder via ['eleitos_holders'] e mede o Gini
#
#   # voto mutuo / Jaccard (usa ctx.aprovacoes):
#   #   similaridade entre linhas (eleitores) da matriz de aprovacoes
# ---------------------------------------------------------------------------

def hhi(ctx: Contexto) -> float:
    """Herfindahl-Hirschman: soma dos quadrados das frações de ctx.shares."""
    raise NotImplementedError("TODO: HHI")  # TODO


def coef_nakamoto(ctx: Contexto, limiar: float = 1/3) -> int:
    """Nº minimo de entidades cujo share acumulado ultrapassa `limiar`."""
    raise NotImplementedError("TODO: coeficiente de Nakamoto sobre ctx.shares")  # TODO


def numero_efetivo(ctx: Contexto) -> float:
    """Nº efetivo (Laakso-Taagepera) = 1 / sum(ctx.shares**2)."""
    raise NotImplementedError("TODO: 1/HHI")  # TODO


def entropia_shannon(ctx: Contexto) -> float:
    """Entropia de Shannon NORMALIZADA de ctx.shares, em [0,1]."""
    raise NotImplementedError("TODO:")  # TODO


def palma(ctx: Contexto) -> float:
    """Razao de Palma: share do topo 10% / share dos 40% inferiores."""
    raise NotImplementedError("TODO: razao de Palma")  # TODO

def permanencia_media(ctx: Contexto) -> float:
    """
    Tempo médio de permanência dos delegados atualmente eleitos.

    Calcula, para cada delegado eleito na rodada atual, há quantas rodadas
    consecutivas ele permanece no comitê.
    """
    historico = ctx.historico

    if not historico:
        return 0.0

    eleitos_atuais = set(ctx.eleitos)
    if not eleitos_atuais:
        return 0.0

    soma = 0

    for delegado in eleitos_atuais:
        permanencia = 1
        # Percorre o histórico (exceto a rodada atual) de trás para frente
        for rodada in reversed(historico[:-1]):
            if delegado in rodada.eleitos:
                permanencia += 1
            else:
                break
        soma += permanencia

    return soma / len(eleitos_atuais)

def diversidade_acumulada(ctx: Contexto) -> float:
    """
    Fração de cadeiras historicamente ocupadas por delegados distintos.

        |⋃ A_t| / (kT)
    """
    if not ctx.historico:
        return 0.0

    ocupantes = set()

    for rodada in ctx.historico:
        ocupantes.update(rodada["eleitos"])

    T = len(ctx.historico)
    k = len(ctx.eleitos)

    return len(ocupantes) / (k * T) if k > 0 else 0.0

# A LLM pode ter esquecido de gerar algum

# Registre aqui as metricas implementadas (o CSV usa estes nomes):
# Casco crie alguma nova métrica lembrar de inserir aqui
METRICAS: dict[str, Callable] = {
    "gini": gini,
    "hhi": hhi,
    "coef_nakamoto": coef_nakamoto,
    "numero_efetivo": numero_efetivo,
    "entropia_shannon": entropia_shannon,
    "palma": palma,
    "permanencia_media": permanencia_media,
    "diversidade": diversidade_acumulada,
}


# ===========================================================================
# 5) MONTE CARLO + CSV (entrada longa: metrica/camada aceitam listas com ';')
# ===========================================================================
def rodar_cenario(cfg, metricas, camadas_alvo, n_runs, seed_base):
    """Roda `n_runs` simulacoes e calcula TODAS as metricas x camadas pedidas
    sobre os MESMOS sorteios. Devolve [(metrica, camada, media, ic95), ...]."""
    acc = {(m, c): [] for m in metricas for c in camadas_alvo}
    for r in range(n_runs):
        rng = np.random.default_rng(seed_base + r)
        b = simular(cfg, rng)
        cam = camadas(b)
        for m in metricas:
            fn = METRICAS[m]
            for c in camadas_alvo:
                sh = cam[c]
                if cfg.frac_exchange > 0:
                    sh = agregar_exchange(sh, cfg.frac_exchange)
                ctx = Contexto(cfg=cfg, camada=c, shares=sh, stakes=b["stakes"],
                               scores=b["scores"], eleitos=b["eleitos"], pool=b["pool"],
                               peso_corte=b["peso_corte"], vota=b["vota"],
                               aprovacoes=b["aprovacoes"], idx_diretos=b["idx_diretos"],
                               blocos=b["blocos"], historico=b["historico"])
                acc[(m, c)].append(float(fn(ctx)))
    saida = []
    for (m, c), am in acc.items():
        media = float(np.mean(am))
        ic95 = float(1.96 * np.std(am, ddof=1) / math.sqrt(n_runs)) if n_runs > 1 else 0.0
        saida.append((m, c, media, ic95))
    return saida


def _lista(celula: str) -> list[str]:
    return [x.strip() for x in str(celula).split(";") if x.strip()]


def _linha_para_config(row: dict):
    base = {}
    for nome in PARAMS:
        if row.get(nome, "") not in ("", None):
            base[nome] = _TIPO[nome](row[nome])
    cfg = Config(**base)
    metricas = _lista(row.get("metrica") or "gini")
    camadas_alvo = _lista(row.get("camada") or "eleito")
    n_runs = int(row.get("n_runs") or 30)
    seed_base = int(row.get("seed_base") or 0)
    return cfg, metricas, camadas_alvo, n_runs, seed_base


def rodar_csv(entrada: str, saida: str) -> list[dict]:
    with open(entrada, newline="", encoding="utf-8") as f:
        cenarios = list(csv.DictReader(f))
    id_exec = datetime.now().strftime("%Y%m%d-%H%M%S")
    linhas = []
    for i, row in enumerate(cenarios, start=1):
        cfg, metricas, camadas_alvo, n_runs, seed_base = _linha_para_config(row)
        for m in metricas:
            if m not in METRICAS:
                raise ValueError(f"metrica '{m}' nao registrada/implementada.")
        for (m, c, media, ic95) in rodar_cenario(cfg, metricas, camadas_alvo, n_runs, seed_base):
            out = {"id_execucao": id_exec, "id_cenario": i}
            out.update({p: getattr(cfg, p) for p in PARAMS})
            out.update({"metrica": m, "camada": c, "media": round(media, 6),
                        "ic95": round(ic95, 6), "n_runs": n_runs, "seed_base": seed_base})
            linhas.append(out)
    _salvar(linhas, saida, CAMPOS_SAIDA)
    return linhas


def _salvar(linhas, caminho, campos):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for L in linhas:
            w.writerow({c: L.get(c, getattr(Config(), c, "")) for c in campos})


def gerar_cenarios(grade, metricas, camadas_alvo, grupo, patologia,
                   n_runs=30, seed_base=0) -> list[dict]:
    """Produto cartesiano de `grade` (param -> lista). Cada combinacao vira UMA
    linha, com metrica/camada gravadas como listas separadas por ';'."""
    chaves = list(grade)
    linhas = []
    for combo in product(*[grade[k] for k in chaves]):
        linha = {"grupo": grupo, "patologia": patologia,
                 "metrica": ";".join(metricas), "camada": ";".join(camadas_alvo),
                 "n_runs": n_runs, "seed_base": seed_base}
        linha.update(dict(zip(chaves, combo)))
        linhas.append(linha)
    return linhas


def salvar_cenarios(linhas, caminho):
    _salvar(linhas, caminho, CAMPOS_ENTRADA)


# ===========================================================================
# 6) EXECUCAO
# ===========================================================================
if __name__ == "__main__":
    if len(sys.argv) == 3:
        out = rodar_csv(sys.argv[1], sys.argv[2])
        print(f"{len(out)} linhas de resultado -> {sys.argv[2]}")
        sys.exit(0)

    # Autoteste do Gini
    assert abs(_gini([1, 1, 1, 1]) - 0.0) < 1e-9
    assert abs(_gini([0, 0, 0, 1]) - 0.75) < 1e-9
    print("Autoteste do Gini: OK")

    # Exemplo G1 (apatia): 2 basicos + turnout, Gini nas 3 camadas
    grade = {"n_holders": [200, 500, 1000],
             "distribuicao": ["pareto", "lognormal"],
             "turnout": [0.1, 0.5, 1.0]}
    cen = gerar_cenarios(grade, metricas=["gini"],
                         camadas_alvo=["stake", "eleito", "produzido"],
                         grupo="G1", patologia="apatia_do_eleitor",
                         n_runs=20, seed_base=0)
    salvar_cenarios(cen, "cenarios_exemplo.csv")
    res = rodar_csv("cenarios_exemplo.csv", "resultados_exemplo.csv")
    print(f"{len(cen)} cenarios -> cenarios_exemplo.csv ; {len(res)} resultados -> resultados_exemplo.csv")