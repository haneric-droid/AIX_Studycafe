"""
04_hypothesis.py — 통계적 가설 검정 (6주차 가설)

H1 (경쟁 밀도)
  H0: 과밀 지역과 비과밀 지역의 독서실 폐업률은 같다.
  H1: 과밀 지역의 독서실 폐업률이 비과밀 지역보다 높다.
  - 과밀 정의 ① 점포 수 기준, ② 10·20대 인구 대비 기준 (같은 분기 안에서 상위 25%)
  - 검정: Welch t-검정(등분산 가정 X), Mann-Whitney U(비모수), 카이제곱(폐업 발생 비율)
H6 (상권 상태)
  H0: 상권변화지표 4개 그룹(상권확장·다이나믹·정체·상권축소)의 폐업률 분포는 같다.
  H1: 적어도 한 그룹은 다르다.
  - 검정: Kruskal-Wallis (+ 참고용 일원분산분석), 유의하면 사후 쌍별 Mann-Whitney(Bonferroni)

분석 단위와 Y
  - 주 분석: 행정동 × 분기 중 분기초 독서실이 1개 이상인 행 (n ≈ 8,500)
    Y = 폐업률(%) = 폐업 수 / 분기초 점포 수 × 100
  - 강건성: 같은 동이 여러 분기 반복되면 관측치가 독립이 아니어서 p-value가 작게 나올 수 있음
    → 동 단위로 기간 전체를 합쳐(동 1개 = 관측치 1개) 같은 검정을 다시 수행
  - 분기별 방향 일관성: 21개 분기 각각에서 과밀 > 비과밀인지 세어 봄
  - 군집-강건 검정(핵심): 폐업률 ~ 그룹 회귀를 동별 군집 표준오차로 추정
      · 비가중: 동·분기 평균 폐업률 비교
      · 점포 수 가중: 합계 폐업률(합계 폐업 / 합계 분기초 점포) 비교 — 점포 1~2개 동의 0%/100% 잡음을 줄임
  ※ Mann-Whitney·카이제곱은 "폐업 0건" 여부에 크게 좌우되는데, 점포가 많은 과밀 동은
    폐업이 1건이라도 생길 확률이 기계적으로 높으므로 해석에 주의

사용법 (작업 폴더에서)
  python src/04_hypothesis.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "panel_dong_quarter.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
Q, DONG = "기준_년분기_코드", "행정동_코드"
ALPHA = 0.05
CHG_ORDER = ["상권확장", "다이나믹", "정체", "상권축소"]

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def verdict(p: float) -> str:
    return "통계적으로 유의미한 차이가 있습니다" if p < ALPHA else "통계적으로 유의미한 차이가 없습니다"


# ---------------------------------------------------------------------------
# 두 집단 비교 (H1)
# ---------------------------------------------------------------------------
def two_group(y1: pd.Series, y0: pd.Series, e1: pd.Series, e0: pd.Series, label: str, unit: str) -> dict:
    """y = 폐업률(%), e = 폐업 발생(0/1). 1 = 과밀, 0 = 비과밀."""
    t_two = stats.ttest_ind(y1, y0, equal_var=False)
    t_one = stats.ttest_ind(y1, y0, equal_var=False, alternative="greater")
    u_two = stats.mannwhitneyu(y1, y0, alternative="two-sided")
    u_one = stats.mannwhitneyu(y1, y0, alternative="greater")
    # 효과 크기: Cohen's d (합동 표준편차), 순위이연상관 r = 2U/(n1 n0) - 1
    sp = np.sqrt(((len(y1) - 1) * y1.var() + (len(y0) - 1) * y0.var()) / (len(y1) + len(y0) - 2))
    d = (y1.mean() - y0.mean()) / sp
    r_rb = 2 * u_two.statistic / (len(y1) * len(y0)) - 1
    # 평균 차이의 95% 신뢰구간 (Welch)
    ci = t_two.confidence_interval(0.95)
    out = {
        "가설": "H1", "비교": label, "분석단위": unit,
        "n_과밀": len(y1), "n_비과밀": len(y0),
        "평균폐업률_과밀": y1.mean(), "평균폐업률_비과밀": y0.mean(),
        "평균차이_%p": y1.mean() - y0.mean(), "차이_95CI_하한": ci.low, "차이_95CI_상한": ci.high,
        "중앙값_과밀": y1.median(), "중앙값_비과밀": y0.median(),
        "t": t_two.statistic, "df": t_two.df, "p_t_양측": t_two.pvalue, "p_t_단측": t_one.pvalue,
        "U": u_two.statistic, "p_U_양측": u_two.pvalue, "p_U_단측": u_one.pvalue,
        "Cohen_d": d, "순위이연상관_r": r_rb,
    }
    if e1 is not None:
        table = np.array([[e1.sum(), len(e1) - e1.sum()], [e0.sum(), len(e0) - e0.sum()]])
        chi2, p_chi, *_ = stats.chi2_contingency(table)
        out.update({"폐업발생비율_과밀": e1.mean(), "폐업발생비율_비과밀": e0.mean(),
                    "chi2": chi2, "p_chi2": p_chi})
    return out


def run_h1(p: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    rows, by_q = [], []
    for col, label in [("과밀_점포수", "점포 수 기준 과밀"), ("과밀_인구대비", "인구 대비 과밀")]:
        g1, g0 = p[p[col] == 1], p[p[col] == 0]
        rows.append(two_group(g1["폐업률"], g0["폐업률"], g1["폐업발생"], g0["폐업발생"], label, "동×분기"))

        # 분기별 방향 일관성
        for q, d in p.groupby(Q):
            a, b = d.loc[d[col] == 1, "폐업률"], d.loc[d[col] == 0, "폐업률"]
            by_q.append({"비교": label, Q: q, "과밀": a.mean(), "비과밀": b.mean(), "과밀이_높음": a.mean() > b.mean()})

    # 강건성: 동 단위 (기간 합계 폐업률, 과밀 = 기간 평균 밀도 상위 25%)
    dong = p.groupby(DONG).agg(폐업=("독서실_폐업수", "sum"), 점포=("독서실_점포수_L1", "sum"),
                               점포수_평균=("독서실_점포수_L1", "mean"),
                               밀도_평균=("독서실_per_1천명_10_20대_L1", "mean"),
                               분기수=(Q, "nunique"))
    dong = dong[dong["분기수"] >= 4]                    # 4개 분기 이상 관측된 동만
    dong["폐업률"] = 100 * dong["폐업"] / dong["점포"]
    for src, label in [("점포수_평균", "점포 수 기준 과밀"), ("밀도_평균", "인구 대비 과밀")]:
        hi = dong[src] > dong[src].quantile(0.75)
        rows.append(two_group(dong.loc[hi, "폐업률"], dong.loc[~hi, "폐업률"], None, None, label, "동(기간 합계)"))
    return rows, pd.DataFrame(by_q)


# ---------------------------------------------------------------------------
# 군집-강건 회귀 검정 (같은 동의 반복 관측을 고려)
# ---------------------------------------------------------------------------
def cluster_test(p: pd.DataFrame, rhs: str, weighted: bool) -> dict:
    """폐업률 ~ rhs, 동별 군집 표준오차. 가중이면 분기초 점포 수로 가중(= 합계 폐업률 비교)."""
    w = p["독서실_점포수_L1"] if weighted else np.ones(len(p))
    m = smf.wls(f"폐업률 ~ {rhs}", data=p, weights=w).fit(
        cov_type="cluster", cov_kwds={"groups": p[DONG].astype("category").cat.codes})
    return m


def run_cluster(p: pd.DataFrame) -> list[dict]:
    rows = []
    for col, label in [("과밀_점포수", "점포 수 기준 과밀"), ("과밀_인구대비", "인구 대비 과밀")]:
        for weighted in (False, True):
            m = cluster_test(p, col, weighted)
            b, se, z = m.params[col], m.bse[col], m.tvalues[col]
            p_one = stats.norm.sf(z)                     # H1: 계수 > 0 (단측)
            ci = m.conf_int().loc[col]
            rows.append({"가설": "H1", "비교": label, "방법": "점포수 가중" if weighted else "비가중",
                         "비과밀_폐업률": m.params["Intercept"], "과밀_차이_%p": b, "SE_군집": se, "z": z,
                         "95CI_하한": ci[0], "95CI_상한": ci[1], "p_양측": m.pvalues[col], "p_단측": p_one,
                         "동_수": p[DONG].nunique()})
    for weighted in (False, True):
        m = cluster_test(p, "C(상권변화지표_L1, Treatment('상권확장'))", weighted)
        names = [n for n in m.params.index if n != "Intercept"]
        wald = m.wald_test(np.eye(len(m.params))[1:], scalar=True)
        rows.append({"가설": "H6", "비교": "상권변화지표 4그룹 (기준: 상권확장)",
                     "방법": "점포수 가중" if weighted else "비가중",
                     "비과밀_폐업률": m.params["Intercept"], "Wald_chi2": float(wald.statistic),
                     "p_양측": float(wald.pvalue),
                     **{f"차이_{n.split('T.')[-1].rstrip(']')}": m.params[n] for n in names},
                     "동_수": p[DONG].nunique()})
    return rows


# ---------------------------------------------------------------------------
# 여러 집단 비교 (H6)
# ---------------------------------------------------------------------------
def run_h6(p: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    groups = [p.loc[p["상권변화지표_L1"] == g, "폐업률"] for g in CHG_ORDER]
    kw = stats.kruskal(*groups)
    an = stats.f_oneway(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    eps2 = (kw.statistic - k + 1) / (n - k)            # 효과 크기 epsilon²
    res = {"가설": "H6", "비교": "상권변화지표 4그룹", "분석단위": "동×분기", "n": n,
           "H": kw.statistic, "p_KW": kw.pvalue, "epsilon2": eps2, "F": an.statistic, "p_ANOVA": an.pvalue}
    for g, s in zip(CHG_ORDER, groups):
        res[f"평균_{g}"] = s.mean()
        res[f"n_{g}"] = len(s)

    # 사후 검정: 쌍별 Mann-Whitney + Bonferroni (참고용으로 항상 계산)
    pairs = []
    m = k * (k - 1) // 2
    for i in range(k):
        for j in range(i + 1, k):
            u = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            pairs.append({"그룹1": CHG_ORDER[i], "그룹2": CHG_ORDER[j], "U": u.statistic,
                          "p": u.pvalue, "p_Bonferroni": min(1, u.pvalue * m)})
    return res, pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# 그래프: 그룹별 평균 폐업률과 95% 신뢰구간
# ---------------------------------------------------------------------------
def mean_ci(s: pd.Series) -> tuple[float, float]:
    half = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
    return s.mean(), half


def plot(p: pd.DataFrame, h1_rows: list[dict], h6: dict, cl: list[dict]) -> None:
    clp = {(r["가설"], r["비교"], r["방법"]): r["p_단측" if r["가설"] == "H1" else "p_양측"] for r in cl}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 2]})
    panels = [("과밀_점포수", "① 점포 수 기준 과밀 (H1)", h1_rows[0]["p_t_단측"], "Welch t 단측"),
              ("과밀_인구대비", "② 인구 대비 과밀 (H1)", h1_rows[1]["p_t_단측"], "Welch t 단측")]
    for ax, (col, title, pv, name) in zip(axes[:2], panels):
        vals = [mean_ci(p.loc[p[col] == g, "폐업률"]) for g in (0, 1)]
        for i, ((m, h), c) in enumerate(zip(vals, [BLUE, ORANGE])):
            ax.errorbar(i, m, yerr=h, fmt="o", color=c, markersize=9, capsize=6, linewidth=2)
            ax.text(i + 0.08, m, f"{m:.2f}%", va="center", fontsize=9, color=TEXT2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["비과밀", "과밀"])
        ax.set_xlim(-0.5, 1.6)
        ax.set_title(title)
        label = "점포 수 기준 과밀" if col == "과밀_점포수" else "인구 대비 과밀"
        ax.text(0.02, 0.04, f"{name}: {fmt_p(pv)}\n동 군집보정·가중: {fmt_p(clp[('H1', label, '점포수 가중')])}",
                transform=ax.transAxes, fontsize=9, color=TEXT2)
    ax = axes[2]
    for i, g in enumerate(CHG_ORDER):
        m, h = mean_ci(p.loc[p["상권변화지표_L1"] == g, "폐업률"])
        ax.errorbar(i, m, yerr=h, fmt="o", color=BLUE, markersize=9, capsize=6, linewidth=2)
        ax.text(i + 0.08, m, f"{m:.2f}%", va="center", fontsize=9, color=TEXT2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(CHG_ORDER)
    ax.set_xlim(-0.5, 3.6)
    ax.set_title("③ 상권변화지표 (H6)")
    h6_key = ("H6", "상권변화지표 4그룹 (기준: 상권확장)", "비가중")
    ax.text(0.02, 0.04, f"Kruskal-Wallis: {fmt_p(h6['p_KW'])}\n동 군집보정 Wald: {fmt_p(clp[h6_key])}",
            transform=ax.transAxes, fontsize=9, color=TEXT2)
    axes[0].set_ylabel("동·분기 평균 폐업률(%)")
    fig.suptitle("그룹별 평균 폐업률과 95% 신뢰구간", x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(FIG / "fig09_가설검정_신뢰구간.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(PANEL, dtype={DONG: str})
    p = p[p["독서실_점포수_L1"] > 0].copy()             # 분기초 점포가 있어야 폐업률 정의 가능
    p["폐업률"] = 100 * p["독서실_폐업수"] / p["독서실_점포수_L1"]
    print(f"분석 대상: {len(p):,}행 ({p[DONG].nunique()}개 동)\n")

    h1_rows, by_q = run_h1(p)
    h6, pairs = run_h6(p)
    cl = run_cluster(p)

    pd.DataFrame(h1_rows).to_csv(TAB / "t08_H1_과밀_검정.csv", index=False, encoding="utf-8-sig")
    by_q.to_csv(TAB / "t09_H1_분기별_방향.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([h6]).to_csv(TAB / "t10_H6_상권변화_검정.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(TAB / "t11_H6_사후검정.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cl).to_csv(TAB / "t12_군집강건_검정.csv", index=False, encoding="utf-8-sig")
    plot(p, h1_rows, h6, cl)

    # ---------------- 콘솔 보고 ----------------
    print("=" * 70)
    print("H1: 과밀 지역의 폐업률이 비과밀 지역보다 높은가?")
    print("=" * 70)
    for r in h1_rows:
        print(f"\n[{r['비교']} / {r['분석단위']}]  n = {r['n_과밀']:,} vs {r['n_비과밀']:,}")
        print(f"  평균 폐업률  과밀 {r['평균폐업률_과밀']:.2f}%  vs  비과밀 {r['평균폐업률_비과밀']:.2f}%"
              f"  (차이 {r['평균차이_%p']:+.2f}%p, 95% CI [{r['차이_95CI_하한']:.2f}, {r['차이_95CI_상한']:.2f}])")
        print(f"  Welch t = {r['t']:.2f}, df = {r['df']:.0f}, 단측 {fmt_p(r['p_t_단측'])} (양측 {fmt_p(r['p_t_양측'])})"
              f"  → {verdict(r['p_t_단측'])}")
        print(f"  Mann-Whitney U = {r['U']:,.0f}, 단측 {fmt_p(r['p_U_단측'])}  → {verdict(r['p_U_단측'])}")
        if "p_chi2" in r:
            print(f"  폐업 발생 비율 {r['폐업발생비율_과밀']:.1%} vs {r['폐업발생비율_비과밀']:.1%},"
                  f" χ² = {r['chi2']:.2f}, {fmt_p(r['p_chi2'])}")
        print(f"  효과 크기: Cohen's d = {r['Cohen_d']:.3f}, 순위이연상관 r = {r['순위이연상관_r']:.3f}")
    for label, d in by_q.groupby("비교"):
        print(f"\n  분기별 방향({label}): 21개 분기 중 {d['과밀이_높음'].sum()}개 분기에서 과밀 > 비과밀")

    print("\n" + "=" * 70)
    print("H6: 상권변화지표에 따라 폐업률이 다른가?")
    print("=" * 70)
    print("  " + ", ".join(f"{g} {h6[f'평균_{g}']:.2f}% (n={h6[f'n_{g}']:,})" for g in CHG_ORDER))
    print(f"  Kruskal-Wallis H = {h6['H']:.2f}, {fmt_p(h6['p_KW'])}, ε² = {h6['epsilon2']:.4f}  → {verdict(h6['p_KW'])}")
    print(f"  (참고) 일원분산분석 F = {h6['F']:.2f}, {fmt_p(h6['p_ANOVA'])}")
    if h6["p_KW"] < ALPHA:
        print("  사후검정(Bonferroni):")
        for _, r in pairs.iterrows():
            print(f"    {r['그룹1']} vs {r['그룹2']}: {fmt_p(r['p_Bonferroni'])}")

    print("\n" + "=" * 70)
    print("군집-강건 검정 (동별 군집 표준오차) — 같은 동의 반복 관측 보정")
    print("=" * 70)
    for r in cl:
        if r["가설"] == "H1":
            print(f"  [H1 {r['비교']} / {r['방법']}] 비과밀 {r['비과밀_폐업률']:.2f}%, 과밀 차이 {r['과밀_차이_%p']:+.2f}%p"
                  f" (95% CI [{r['95CI_하한']:.2f}, {r['95CI_상한']:.2f}]), z = {r['z']:.2f},"
                  f" 단측 {fmt_p(r['p_단측'])}  → {verdict(r['p_단측'])}")
        else:
            diffs = ", ".join(f"{k[3:]} {v:+.2f}%p" for k, v in r.items() if k.startswith("차이_"))
            print(f"  [H6 / {r['방법']}] 상권확장 {r['비과밀_폐업률']:.2f}% 기준: {diffs};"
                  f" Wald χ² = {r['Wald_chi2']:.2f}, {fmt_p(r['p_양측'])}  → {verdict(r['p_양측'])}")
    print("\n결과 표:", TAB)


if __name__ == "__main__":
    main()
