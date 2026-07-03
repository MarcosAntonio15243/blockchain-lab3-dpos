"""
Analise e visualizacao de resultados.csv - Lab 03 DPoS (G11 - Entrincheiramento)
=================================================================================

Le resultados.csv (saida do simulador) e gera as figuras usadas no relatorio:

  fig1_hipotese_central.png      efeito de vantagem_incumbencia x n_rodadas
                                  sobre as 4 metricas do grupo (o resultado
                                  central: entrincheiramento reduz rotatividade
                                  e aumenta persistencia/permanencia)
  fig2_n_candidatos.png          efeito do numero de candidatos (agora <=
                                  n_holders em alguns cenarios, entao filtra
                                  de fato o pool de candidatos)
  fig3_n_holders.png             efeito do numero de detentores (basico)
  fig4_invariancia_camadas.png   evidencia de que estas metricas NAO variam
                                  por camada (stake/eleito/produzido), ao
                                  contrario das metricas de concentracao
  fig5_heatmap_rotatividade.png  resumo compacto (vantagem x n_rodadas)
  fig6_correlacao_metricas.png   rotatividade vs persistencia (relacao entre
                                  metricas da mesma familia)
  tabela_resumo.csv              deltas numericos (baixa vs alta incumbencia)
                                  prontos para citar no texto do relatorio

Uso:
    python analise_resultados.py [resultados.csv] [pasta_saida]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0) Paleta e estilo (dataviz skill: ordem categorica fixa, ramp sequencial
#    para grandezas ordinais, tinta/grid recessivos, marcas finas)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# categorica (ordem fixa, nunca ciclada): usada p/ identidade (ex.: camada)
CAT = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CAMADA_COLOR = {"stake": CAT["blue"], "eleito": CAT["aqua"], "produzido": CAT["yellow"]}
CAMADA_ORDEM = ["stake", "eleito", "produzido"]

# sequencial azul (grandezas ordinais: n_rodadas, vantagem_incumbencia)
SEQ_BLUE_3 = ["#86b6ef", "#2a78d6", "#104281"]  # steps 250 / 450 / 650

METRICAS_LABEL = {
    "rotatividade": "Taxa de rotatividade",
    "persistencia": "Persistência",
    "permanencia_media": "Permanência média (rodadas)",
    "diversidade": "Diversidade acumulada",
}
METRICAS_ORDEM = ["rotatividade", "persistencia", "permanencia_media", "diversidade"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "font.family": "sans-serif",
    "legend.frameon": False,
})


def carregar(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    for c in ["vantagem_incumbencia", "n_rodadas", "n_holders", "n_candidatos",
              "tamanho_comite", "media", "ic95"]:
        df[c] = pd.to_numeric(df[c])
    return df


def _errorbar(ax, x, y, yerr, color, label, marker="o"):
    ax.errorbar(
        x, y, yerr=yerr, color=color, label=label, marker=marker,
        markersize=6, linewidth=2, capsize=3, capthick=1.2,
        markeredgecolor=SURFACE, markeredgewidth=0.8,
    )


# ---------------------------------------------------------------------------
# Fig 1 - hipotese central: vantagem_incumbencia x n_rodadas, 4 metricas
# ---------------------------------------------------------------------------
def fig1_hipotese_central(df: pd.DataFrame, out: Path,
                           n_holders=1000, n_candidatos=150, tamanho_comite=21,
                           camada="eleito"):
    sub = df[(df.n_holders == n_holders) & (df.n_candidatos == n_candidatos) &
             (df.tamanho_comite == tamanho_comite) & (df.camada == camada)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    rodadas = sorted(sub.n_rodadas.unique())
    vants = sorted(sub.vantagem_incumbencia.unique())

    for ax, metrica in zip(axes.flat, METRICAS_ORDEM):
        m = sub[sub.metrica == metrica].sort_values("vantagem_incumbencia")
        for cor, nr in zip(SEQ_BLUE_3, rodadas):
            r = m[m.n_rodadas == nr]
            _errorbar(ax, r.vantagem_incumbencia, r.media, r.ic95, cor, f"{int(nr)} rodadas")
        ax.set_title(METRICAS_LABEL[metrica], color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xticks(vants)
        ax.set_xlim(vants[0] - 0.08, vants[-1] + 0.08)

    for ax in axes[1, :]:
        ax.set_xlabel("vantagem_incumbencia")
    axes[0, 0].legend(loc="upper right", title="n_rodadas", fontsize=9)

    fig.suptitle(
        f"Efeito da vantagem de incumbência (G6) — n_holders={n_holders}, "
        f"n_candidatos={n_candidatos}, k={tamanho_comite}, camada={camada}\n"
        "barras = IC95% sobre 30 sementes",
        fontsize=11, color=INK_SECONDARY, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out / "fig1_hipotese_central.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 - efeito de n_candidatos (agora pode filtrar o pool de fato, pois
# passou a incluir valores <= n_holders)
# ---------------------------------------------------------------------------
def fig2_n_candidatos(df: pd.DataFrame, out: Path,
                       n_holders=1000, tamanho_comite=21, n_rodadas=10,
                       camada="eleito"):
    sub = df[(df.n_holders == n_holders) & (df.tamanho_comite == tamanho_comite) &
             (df.n_rodadas == n_rodadas) & (df.camada == camada)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    vants = sorted(sub.vantagem_incumbencia.unique())

    for ax, metrica in zip(axes.flat, METRICAS_ORDEM):
        m = sub[sub.metrica == metrica].sort_values("n_candidatos")
        for cor, v in zip(SEQ_BLUE_3, vants):
            r = m[m.vantagem_incumbencia == v]
            _errorbar(ax, r.n_candidatos, r.media, r.ic95, cor, f"vant.={v:g}")
        ax.set_title(METRICAS_LABEL[metrica], color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xticks(sorted(sub.n_candidatos.unique()))

    for ax in axes[1, :]:
        ax.set_xlabel("n_candidatos")
    axes[0, 0].legend(loc="upper right", title="vantagem_incumbencia", fontsize=9)

    fig.suptitle(
        f"Efeito de n_candidatos — n_holders={n_holders}, "
        f"k={tamanho_comite}, n_rodadas={n_rodadas}, camada={camada}\n"
        "barras = IC95% sobre 30 sementes",
        fontsize=11, color=INK_SECONDARY, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out / "fig2_n_candidatos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 - efeito de n_holders (parametro basico)
# ---------------------------------------------------------------------------
def fig3_n_holders(df: pd.DataFrame, out: Path,
                    n_candidatos=150, tamanho_comite=21, n_rodadas=10,
                    camada="eleito"):
    sub = df[(df.n_candidatos == n_candidatos) & (df.tamanho_comite == tamanho_comite) &
             (df.n_rodadas == n_rodadas) & (df.camada == camada)]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    vants = sorted(sub.vantagem_incumbencia.unique())

    for ax, metrica in zip(axes.flat, METRICAS_ORDEM):
        m = sub[sub.metrica == metrica].sort_values("n_holders")
        for cor, v in zip(SEQ_BLUE_3, vants):
            r = m[m.vantagem_incumbencia == v]
            _errorbar(ax, r.n_holders, r.media, r.ic95, cor, f"vant.={v:g}")
        ax.set_title(METRICAS_LABEL[metrica], color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xticks(sorted(sub.n_holders.unique()))

    for ax in axes[1, :]:
        ax.set_xlabel("n_holders")
    axes[0, 0].legend(loc="upper right", title="vantagem_incumbencia", fontsize=9)

    fig.suptitle(
        f"Efeito de n_holders — n_candidatos={n_candidatos}, k={tamanho_comite}, "
        f"n_rodadas={n_rodadas}, camada={camada}\nbarras = IC95% sobre 30 sementes",
        fontsize=11, color=INK_SECONDARY, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out / "fig3_n_holders.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 - invariancia por camada (achado: gap == 0 para estas metricas)
# ---------------------------------------------------------------------------
def fig4_invariancia_camadas(df: pd.DataFrame, out: Path,
                              n_holders=1000, n_candidatos=150, tamanho_comite=21,
                              n_rodadas=10):
    sub = df[(df.n_holders == n_holders) & (df.n_candidatos == n_candidatos) &
             (df.tamanho_comite == tamanho_comite) & (df.n_rodadas == n_rodadas)]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=False)
    vants = sorted(sub.vantagem_incumbencia.unique())
    width = 0.25
    x = np.arange(len(vants))

    for ax, metrica in zip(axes, METRICAS_ORDEM):
        m = sub[sub.metrica == metrica]
        for i, camada in enumerate(CAMADA_ORDEM):
            r = m[m.camada == camada].sort_values("vantagem_incumbencia")
            ax.bar(x + (i - 1) * width, r.media, width=width, yerr=r.ic95,
                   color=CAMADA_COLOR[camada], label=camada, capsize=2,
                   edgecolor=SURFACE, linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in vants])
        ax.set_title(METRICAS_LABEL[metrica], fontsize=10, loc="left")
        ax.set_xlabel("vantagem_incumbencia")

    axes[0].legend(loc="upper right", title="camada", fontsize=8)
    fig.suptitle(
        "As barras das 3 camadas se sobrepõem: rotatividade, persistência, "
        "permanência e diversidade dependem apenas da identidade dos eleitos ao\n"
        "longo das rodadas (ctx.historico) — não da distribuição de peso "
        "(ctx.shares) — logo não há 'gap' entre stake / eleito / produzido aqui.",
        fontsize=9.5, color=INK_SECONDARY, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(out / "fig4_invariancia_camadas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 - heatmap resumo (rotatividade) vantagem x n_rodadas
# ---------------------------------------------------------------------------
def fig5_heatmap_rotatividade(df: pd.DataFrame, out: Path,
                               n_holders=1000, n_candidatos=150, tamanho_comite=21,
                               camada="eleito", metrica="rotatividade"):
    sub = df[(df.n_holders == n_holders) & (df.n_candidatos == n_candidatos) &
             (df.tamanho_comite == tamanho_comite) & (df.camada == camada) &
             (df.metrica == metrica)]
    piv = sub.pivot_table(index="n_rodadas", columns="vantagem_incumbencia", values="media")
    piv = piv.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{int(v)}" for v in piv.index])
    ax.set_xlabel("vantagem_incumbencia")
    ax.set_ylabel("n_rodadas")
    ax.grid(False)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            txt_color = SURFACE if val > piv.values.max() * 0.55 else INK_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    ax.set_title(
        f"{METRICAS_LABEL[metrica]} — n_holders={n_holders}, k={tamanho_comite}",
        fontsize=10, loc="left",
    )
    fig.tight_layout()
    fig.savefig(out / "fig5_heatmap_rotatividade.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 - correlacao entre metricas (rotatividade x persistencia)
# ---------------------------------------------------------------------------
def fig6_correlacao_metricas(df: pd.DataFrame, out: Path, camada="eleito"):
    piv = df[df.camada == camada].pivot_table(
        index=["n_holders", "n_candidatos", "tamanho_comite", "n_rodadas", "vantagem_incumbencia"],
        columns="metrica", values="media",
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    vants = sorted(piv.vantagem_incumbencia.unique())
    for cor, v in zip(SEQ_BLUE_3, vants):
        r = piv[piv.vantagem_incumbencia == v]
        ax.scatter(r.rotatividade, r.persistencia, color=cor, s=28, alpha=0.85,
                   edgecolor=SURFACE, linewidth=0.5, label=f"vant.={v:g}")

    corr = piv["rotatividade"].corr(piv["persistencia"])
    ax.set_xlabel("Taxa de rotatividade")
    ax.set_ylabel("Persistência")
    ax.set_title(f"Rotatividade vs. persistência (camada={camada}, r={corr:.2f})",
                 fontsize=10.5, loc="left")
    ax.legend(title="vantagem_incumbencia", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig6_correlacao_metricas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tabela resumo: delta baixa (min) vs alta (max) incumbencia testada, por metrica
# ---------------------------------------------------------------------------
def tabela_resumo(df: pd.DataFrame, out: Path, camada="eleito"):
    vant_baixa = df.vantagem_incumbencia.min()
    vant_alta = df.vantagem_incumbencia.max()
    linhas = []
    for metrica in METRICAS_ORDEM:
        sub = df[(df.metrica == metrica) & (df.camada == camada)]
        agg = sub.groupby("vantagem_incumbencia")["media"].mean()
        baixa, alta = agg.get(vant_baixa), agg.get(vant_alta)
        linhas.append({
            "metrica": metrica,
            f"media_vant_{vant_baixa:g}": round(baixa, 4) if baixa is not None else None,
            f"media_vant_{vant_alta:g}": round(alta, 4) if alta is not None else None,
            "delta_absoluto": round(alta - baixa, 4) if None not in (baixa, alta) else None,
            "delta_percentual": round(100 * (alta - baixa) / baixa, 1)
            if baixa not in (None, 0) else None,
        })
    tab = pd.DataFrame(linhas)
    tab.to_csv(out / "tabela_resumo.csv", index=False)
    return tab


def main():
    caminho_csv = sys.argv[1] if len(sys.argv) > 1 else "resultados.csv"
    pasta_saida = Path(sys.argv[2] if len(sys.argv) > 2 else "figuras")
    pasta_saida.mkdir(exist_ok=True)

    df = carregar(caminho_csv)
    print(f"{len(df)} linhas carregadas de {caminho_csv}")

    fig1_hipotese_central(df, pasta_saida)
    fig2_n_candidatos(df, pasta_saida)
    fig3_n_holders(df, pasta_saida)
    fig4_invariancia_camadas(df, pasta_saida)
    fig5_heatmap_rotatividade(df, pasta_saida)
    fig6_correlacao_metricas(df, pasta_saida)
    tab = tabela_resumo(df, pasta_saida)

    print(f"Figuras salvas em {pasta_saida}/")
    print(f"\nResumo (delta vantagem_incumbencia {df.vantagem_incumbencia.min():g} -> "
          f"{df.vantagem_incumbencia.max():g}, camada=eleito):")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
