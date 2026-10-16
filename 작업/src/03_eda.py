"""
03_eda.py — 탐색적 분석(EDA)

입력 : data/raw/seoul/상권분석_점포_행정동_독서실.csv  (분기별 추이, 2021Q1 포함)
       data/processed/panel_dong_quarter.csv          (행정동×분기 패널)
출력 : outputs/figures/fig01~fig07 *.png
       outputs/tables/*.csv

폐업률 집계 기준
  여러 동·분기를 묶어서 폐업률을 계산할 때는
  "합계 폐업 수 / 합계 직전 분기 점포 수(분기 초 점포 수)"를 쓴다.
  → 분기 초에 영업 중이던 점포 중 몇 %가 그 분기에 폐업했는지 (위험 집단 기준)

사용법 (작업 폴더에서)
  python src/03_eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "seoul"
PANEL = ROOT / "data" / "processed" / "panel_dong_quarter.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
Q = "기준_년분기_코드"

# ---------------------------------------------------------------------------
# 그래프 스타일 (dataviz 기본 팔레트, 라이트 모드)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # 범주형 1~3번 슬롯
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # 순차(파랑)
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"                         # 발산(파랑↔빨강)

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9,
})


def qlabel(q: int) -> str:
    return f"{str(q)[2:4]}Q{q % 10}"


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  저장:", name)


def pooled_rate(df: pd.DataFrame) -> float:
    """합계 폐업 수 / 합계 분기초 점포 수 (%)"""
    base = df["독서실_점포수_L1"].sum()
    return 100 * df["독서실_폐업수"].sum() / base if base else np.nan


# ---------------------------------------------------------------------------
# 1. 분기별 추이 (서울 전체)
# ---------------------------------------------------------------------------
def fig_trend() -> None:
    raw = pd.read_csv(RAW / "상권분석_점포_행정동_독서실.csv")
    t = raw.groupby(Q)[["전체_점포_수", "개업_점포_수", "폐업_점포_수"]].sum()
    t.columns = ["점포수", "개업수", "폐업수"]
    t["분기초_점포수"] = t["점포수"].shift(1)
    t["폐업률_%"] = 100 * t["폐업수"] / t["분기초_점포수"]
    t["개업률_%"] = 100 * t["개업수"] / t["분기초_점포수"]
    t.to_csv(TAB / "t01_분기별_추이.csv", encoding="utf-8-sig")

    x = np.arange(len(t))
    labels = [qlabel(q) for q in t.index]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2, 1]})

    ax = axes[0]
    ax.plot(x, t["점포수"], color=BLUE, marker="o", markersize=4)
    ax.set_title("① 서울 독서실·스터디카페 점포 수")
    ax.set_ylabel("점포 수(개)")
    for i in [0, len(t) - 1]:
        ax.annotate(f"{t['점포수'].iloc[i]:,.0f}", (x[i], t["점포수"].iloc[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=TEXT)

    ax = axes[1]
    ax.plot(x, t["개업수"], color=BLUE, marker="o", markersize=4, label="개업")
    ax.plot(x, t["폐업수"], color=ORANGE, marker="o", markersize=4, label="폐업")
    ax.set_title("② 분기별 개업·폐업 점포 수")
    ax.set_ylabel("점포 수(개)")
    ax.legend(loc="upper left", ncol=2)
    for col, c in [("개업수", BLUE), ("폐업수", ORANGE)]:
        ax.annotate(col[:2], (x[-1], t[col].iloc[-1]), textcoords="offset points", xytext=(8, 0),
                    va="center", fontsize=9, color=TEXT2)
    peak = t["폐업수"].idxmax()
    ax.annotate(f"폐업 최고 {t.loc[peak, '폐업수']:.0f}개 ({qlabel(peak)})",
                (list(t.index).index(peak), t.loc[peak, "폐업수"]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color=TEXT)

    ax = axes[2]
    ax.plot(x[1:], t["폐업률_%"].iloc[1:], color=ORANGE, marker="o", markersize=4)
    ax.axhline(t["폐업률_%"].mean(), color=MUTED, linewidth=1, linestyle="--")
    ax.text(x[1], t["폐업률_%"].mean(), f" 평균 {t['폐업률_%'].mean():.1f}%", va="bottom", fontsize=9, color=TEXT2)
    ax.set_title("③ 분기 폐업률 (폐업 수 / 분기초 점포 수)")
    ax.set_ylabel("폐업률(%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    fig.tight_layout()
    save(fig, "fig01_분기별_추이.png")


# ---------------------------------------------------------------------------
# 2. 자치구별 폐업률 + 3. 자치구 × 연도 히트맵
# ---------------------------------------------------------------------------
def fig_gu(p: pd.DataFrame) -> None:
    g = p.groupby("자치구명").apply(lambda d: pd.Series({
        "폐업률_%": pooled_rate(d),
        "평균_점포수": d.groupby(Q)["독서실_점포수_L1"].sum().mean(),
        "폐업수_합계": d["독서실_폐업수"].sum(),
    }), include_groups=False).sort_values("폐업률_%")
    g.to_csv(TAB / "t02_자치구별_폐업률.csv", encoding="utf-8-sig")
    seoul = pooled_rate(p)

    fig, ax = plt.subplots(figsize=(9, 8.5))
    y = np.arange(len(g))
    ax.barh(y, g["폐업률_%"], color=BLUE, height=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axvline(seoul, color=TEXT2, linewidth=1, linestyle="--", zorder=1)
    ax.text(seoul, len(g) - 0.3, f" 서울 평균 {seoul:.2f}%", fontsize=9, color=TEXT2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(g.index)
    for i, (v, n) in enumerate(zip(g["폐업률_%"], g["평균_점포수"])):
        ax.text(v + 0.05, i, f"{v:.2f}%  (점포 {n:.0f})", va="center", fontsize=8.5, color=TEXT2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1), zorder=3)
    ax.set_xlim(0, g["폐업률_%"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("분기 평균 폐업률(%) — 2021Q2~2026Q2 합계 기준")
    ax.set_title("자치구별 독서실·스터디카페 분기 폐업률")
    fig.tight_layout()
    save(fig, "fig02_자치구별_폐업률.png")

    # 자치구 × 연도
    h = p.groupby(["자치구명", "연도"]).apply(pooled_rate, include_groups=False).unstack()
    h = h.loc[g.index[::-1]]
    h.to_csv(TAB / "t03_자치구_연도별_폐업률.csv", encoding="utf-8-sig")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(h.values, cmap=cmap, aspect="auto")
    ax.set_xticks(range(h.shape[1]))
    ax.set_xticklabels([f"{c}" + ("(Q2~)" if c == 2021 else "(~Q2)" if c == 2026 else "") for c in h.columns])
    ax.set_yticks(range(h.shape[0]))
    ax.set_yticklabels(h.index)
    ax.grid(False)
    vmax = np.nanmax(h.values)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > vmax * 0.6 else TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("분기 평균 폐업률(%)", color=TEXT2)
    cb.outline.set_visible(False)
    ax.set_title("자치구 × 연도별 분기 평균 폐업률")
    fig.tight_layout()
    save(fig, "fig03_자치구_연도_히트맵.png")


# ---------------------------------------------------------------------------
# 4. 변수 분포
# ---------------------------------------------------------------------------
DIST_VARS = [
    ("독서실_점포수_L1", "독서실 점포 수", False),
    ("독서실_per_1천명_10_20대_L1", "10·20대 1천명당 독서실 수", True),
    ("독서실_개업수_L1", "직전 분기 개업 수", False),
    ("상주_10_20대_L1", "10·20대 상주인구", True),
    ("유동_10_20대_L1", "10·20대 유동인구", True),
    ("학원_일반교습수_L1", "일반교습학원 수", True),
    ("교육지출_총액_L1", "교육지출 총액(원)", True),
    ("아파트_평균시가_억_L1", "아파트 평균 시가(억원)", True),
]


def fig_dist(p: pd.DataFrame) -> None:
    desc = p[["독서실_폐업수", "독서실_폐업률"] + [v for v, *_ in DIST_VARS]].describe(
        percentiles=[.01, .25, .5, .75, .99]).T
    desc["왜도"] = p[desc.index].skew()
    desc.to_csv(TAB / "t04_기술통계.csv", encoding="utf-8-sig")

    # Y 분포
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    vc = p["독서실_폐업수"].clip(upper=5).astype(int).value_counts().sort_index()
    ax = axes[0]
    ax.bar(vc.index, vc.values, color=ORANGE, width=0.7, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(vc.index)
    ax.set_xticklabels([str(i) if i < 5 else "5+" for i in vc.index])
    for i, v in zip(vc.index, vc.values):
        ax.text(i, v, f"{v:,}\n({v / len(p):.0%})", ha="center", va="bottom", fontsize=8.5, color=TEXT2)
    ax.set_ylim(0, vc.max() * 1.2)
    ax.set_title("① 동·분기별 폐업 수")
    ax.set_xlabel("폐업 점포 수")
    ax.set_ylabel("동·분기 수")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    r = p.loc[p["독서실_폐업률"] > 0, "독서실_폐업률"]
    ax.hist(r, bins=30, color=ORANGE, edgecolor=SURFACE, linewidth=1)
    ax.set_title("② 폐업률 분포 (폐업이 있는 동·분기)")
    ax.set_xlabel("폐업률(%)")
    ax.set_ylabel("동·분기 수")
    fig.tight_layout()
    save(fig, "fig04_종속변수_분포.png")

    # X 분포
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, (v, title, use_log) in zip(axes.flat, DIST_VARS):
        s = p[v].dropna()
        if use_log:
            s = np.log10(s[s > 0])
            title += " (log10)"
        ax.hist(s, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(labelsize=8)
    fig.suptitle("주요 설명변수 분포 (직전 분기 값)", x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    save(fig, "fig05_설명변수_분포.png")


# ---------------------------------------------------------------------------
# 5. 상관관계 (스피어만: 치우친 분포·이상치에 강함)
# ---------------------------------------------------------------------------
CORR_VARS = {
    "독서실_폐업수": "폐업 수(Y)", "독서실_폐업률": "폐업률(Y)",
    "독서실_점포수_L1": "독서실 점포수", "독서실_per_1천명_10_20대_L1": "인구대비 독서실",
    "독서실_개업수_L1": "직전 개업수", "독서실_프랜차이즈비율_L1": "프랜차이즈 비율",
    "학원_일반교습수_L1": "일반교습학원", "상주_10_20대_L1": "상주 10·20대",
    "직장_총_L1": "직장인구", "유동_10_20대_L1": "유동 10·20대", "학교_초중고수_L1": "초중고 수",
    "대학수_L1": "대학 수", "지하철역수_L1": "지하철역", "교육지출_총액_L1": "교육지출",
    "아파트_평균시가_억_L1": "아파트 시가", "상권_폐업개월평균_L1": "상권 폐업개월",
}


def fig_corr(p: pd.DataFrame) -> None:
    c = p[list(CORR_VARS)].rename(columns=CORR_VARS).corr(method="spearman")
    c.to_csv(TAB / "t05_스피어만_상관계수.csv", encoding="utf-8-sig")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    fig, ax = plt.subplots(figsize=(11, 9.5))
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    im = ax.imshow(np.ma.masked_where(mask, c.values), cmap=cmap, vmin=-1, vmax=1)
    n = len(c)
    ax.set_xticks(range(n))
    ax.set_xticklabels(c.columns, rotation=55, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(c.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(i + 1):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else TEXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("스피어만 상관계수", color=TEXT2)
    cb.outline.set_visible(False)
    ax.set_title("변수 간 상관관계 (스피어만)")
    fig.tight_layout()
    save(fig, "fig06_상관관계_히트맵.png")


# ---------------------------------------------------------------------------
# 6. 설명변수 5분위별 폐업률 (가설 방향 미리 보기)
# ---------------------------------------------------------------------------
BIN_VARS = [
    ("독서실_점포수_L1", "독서실 점포 수 (H1)"),
    ("독서실_per_1천명_10_20대_L1", "인구 대비 독서실 (H1)"),
    ("독서실_개업수_L1", "직전 분기 개업 수 (H2)"),
    ("상주_10_20대_L1", "10·20대 상주인구 (H3)"),
    ("유동_10_20대_L1", "10·20대 유동인구 (H3)"),
    ("학원_일반교습수_L1", "일반교습학원 수 (H4)"),
    ("아파트_평균시가_억_L1", "아파트 평균 시가 (H5)"),
    ("교육지출_총액_L1", "교육지출 총액"),
]


def fig_bins(p: pd.DataFrame) -> None:
    d = p[p["독서실_점포수_L1"] > 0]
    seoul = pooled_rate(d)
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, (v, title) in zip(axes.flat, BIN_VARS):
        b = pd.qcut(d[v].rank(method="first"), 5, labels=["1(낮음)", "2", "3", "4", "5(높음)"])
        r = d.groupby(b, observed=True).apply(pooled_rate, include_groups=False)
        rng = d.groupby(b, observed=True)[v].agg(["min", "max"])
        for k in r.index:
            rows.append({"변수": v, "분위": k, "최소": rng.loc[k, "min"], "최대": rng.loc[k, "max"], "폐업률_%": r[k]})
        ax.bar(range(5), r.values, color=BLUE, width=0.7, edgecolor=SURFACE, linewidth=2)
        ax.axhline(seoul, color=MUTED, linewidth=1, linestyle="--")
        for i, val in enumerate(r.values):
            ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, color=TEXT2)
        ax.set_xticks(range(5))
        ax.set_xticklabels(r.index, fontsize=8)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("분기 폐업률(%)")
    fig.suptitle(f"설명변수 5분위별 독서실 폐업률  (점선 = 전체 평균 {seoul:.2f}%)",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    save(fig, "fig07_분위별_폐업률.png")
    pd.DataFrame(rows).to_csv(TAB / "t06_분위별_폐업률.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 7. 범주형 변수: 과밀 여부, 상권변화지표
# ---------------------------------------------------------------------------
def fig_groups(p: pd.DataFrame) -> None:
    d = p[p["독서실_점포수_L1"] > 0]
    groups = [
        ("과밀_점포수", {0: "비과밀", 1: "과밀"}, "① 점포 수 기준 과밀 (H1)"),
        ("과밀_인구대비", {0: "비과밀", 1: "과밀"}, "② 인구 대비 과밀 (H1)"),
        ("상권변화지표_L1", None, "③ 상권변화지표 (H6)"),
    ]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 2]})
    for ax, (col, names, title) in zip(axes, groups):
        r = d.groupby(col).apply(lambda x: pd.Series({"폐업률": pooled_rate(x), "n": len(x)}), include_groups=False)
        if names:
            r.index = [names[i] for i in r.index]
        else:
            r = r.loc[["상권확장", "다이나믹", "정체", "상권축소"]]
        for k, row in r.iterrows():
            rows.append({"변수": col, "그룹": k, "폐업률_%": row["폐업률"], "동·분기 수": int(row["n"])})
        ax.bar(range(len(r)), r["폐업률"], color=BLUE, width=0.6, edgecolor=SURFACE, linewidth=2)
        for i, (val, n) in enumerate(zip(r["폐업률"], r["n"])):
            ax.text(i, val, f"{val:.2f}%\n(n={int(n):,})", ha="center", va="bottom", fontsize=8.5, color=TEXT2)
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels(r.index)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("분기 폐업률(%)")
    axes[0].set_ylim(0, max(x["폐업률_%"] for x in rows) * 1.3)
    fig.suptitle("그룹별 독서실 폐업률", x=0.01, ha="left", fontsize=13, fontweight="bold", color=TEXT)
    fig.tight_layout()
    save(fig, "fig08_그룹별_폐업률.png")
    pd.DataFrame(rows).to_csv(TAB / "t07_그룹별_폐업률.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(PANEL, dtype={"행정동_코드": str})
    print(f"패널 {len(p):,}행 로드")
    fig_trend()
    fig_gu(p)
    fig_dist(p)
    fig_corr(p)
    fig_bins(p)
    fig_groups(p)
    print("완료:", FIG)


if __name__ == "__main__":
    main()
