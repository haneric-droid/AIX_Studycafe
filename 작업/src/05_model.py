"""
05_model.py — 다변량 회귀 모형: 어떤 요인이 독서실 폐업률을 움직이는가

분석 대상: 분기초 독서실이 1개 이상인 행정동 × 분기 (2021Q2~2026Q2)
공통 설정: X는 모두 직전 분기(t-1) 값, 표준오차는 동별 군집(cluster) 표준오차

모형
  M1 음이항 회귀 (폐업 수, offset = log 분기초 점포 수) + 분기 FE + 자치구 FE
       → 지역 간 차이 + 시간 변화를 모두 사용. 계수 = 발생률비(IRR, 점포당 폐업률 배수)
  M2 포아송 회귀 + 동 FE + 분기 FE (offset 동일)
       → 같은 동 안에서 시간에 따라 변한 부분만 사용 (동 고유 특성 통제).
         동 FE가 있는 포아송은 과대산포가 있어도 군집 표준오차와 함께 쓰면 일관된 추정량이라
         동 FE 음이항(모수 420여 개, 수렴 불안정) 대신 표준적으로 사용한다.
         기간 내내 폐업이 0건인 동은 동 FE 추정이 불가능해 자동 제외된다.
  M3 로지스틱 회귀 (폐업 발생 0/1) + 분기 FE + 자치구 FE
       → 계수 = 오즈비(OR). 4주차 시뮬레이터의 "폐업 확률" 계산에 바로 쓸 수 있는 형태
  (조건부 로지스틱 동 FE는 수렴 실패로 제외 — 동 FE 분석은 M2가 담당)
  강건성: M1을 2021년(코로나 영향기) 제외 / 직장인구 추가 버전으로 재추정

변수 변환 (해석하기 쉽게)
  연속형은 log2 → "그 변수가 2배가 될 때" 폐업률(또는 오즈)이 몇 배가 되는지로 읽음
  log2(점포수) 계수는 offset과 함께 있으므로, 1보다 크면 "점포가 많을수록 점포당 폐업률이 높다"
  = 경쟁 효과(H1). 상주 10·20대 인구를 같이 넣으므로 인구 대비 밀도의 효과와 같은 의미.

사용법 (작업 폴더에서)
  python src/05_model.py
"""

import os
import sys
import warnings
from pathlib import Path

# 이 PC에서는 멀티스레드 BLAS가 오히려 수십 배 느려서 단일 스레드로 고정 (numpy import 전에 설정)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "panel_dong_quarter.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
Q, DONG = "기준_년분기_코드", "행정동_코드"

# 설명변수: (모형 변수명, 표시 이름, 가설)
X_VARS = [
    ("lg_점포", "독서실 점포 수 (2배)", "H1 경쟁"),
    ("lg_개업", "직전 분기 개업 수 (1+x, 2배)", "H2 출점"),
    ("lg_상주1020", "10·20대 상주인구 (2배)", "H3 수요"),
    ("lg_유동1020", "10·20대 유동인구 (2배)", "H3 수요"),
    ("lg_학원", "일반교습학원 수 (1+x, 2배)", "H4 학원가"),
    ("lg_시가", "아파트 평균 시가 (2배)", "H5 고정비"),
    ("프랜차이즈_10p", "프랜차이즈 비율 (+10%p)", "통제"),
    ("지하철역수", "지하철역 수 (+1)", "통제"),
    ("대학수", "대학 수 (+1)", "통제"),
    ("chg_다이나믹", "상권: 다이나믹 (vs 확장)", "H6 상권"),
    ("chg_정체", "상권: 정체 (vs 확장)", "H6 상권"),
    ("chg_상권축소", "상권: 축소 (vs 확장)", "H6 상권"),
]
XN = [v for v, *_ in X_VARS]
LABEL = {v: l for v, l, _ in X_VARS}
HYP = {v: h for v, _, h in X_VARS}

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # 범주형 1~3번 (3개까지 전 쌍 검증된 순서)
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# 1. 데이터 준비
# ---------------------------------------------------------------------------
def prepare() -> pd.DataFrame:
    p = pd.read_csv(PANEL, dtype={DONG: str, "자치구_코드": str})
    p = p[p["독서실_점포수_L1"] > 0].copy()

    # 극단값 완화: 상주 10·20대 인구 하위 1% 절단 (둔촌1동 재건축 등)
    lo = p["상주_10_20대_L1"].quantile(0.01)
    p["상주1020_w"] = p["상주_10_20대_L1"].clip(lower=lo)

    p["lg_점포"] = np.log2(p["독서실_점포수_L1"])
    p["lg_개업"] = np.log2(1 + p["독서실_개업수_L1"])
    p["lg_상주1020"] = np.log2(p["상주1020_w"])
    p["lg_유동1020"] = np.log2(p["유동_10_20대_L1"])
    p["lg_학원"] = np.log2(1 + p["학원_일반교습수_L1"])
    p["lg_시가"] = np.log2(p["아파트_평균시가_억_L1"])
    p["lg_직장"] = np.log2(p["직장_총_L1"])
    p["프랜차이즈_10p"] = p["독서실_프랜차이즈비율_L1"] * 10
    p["지하철역수"] = p["지하철역수_L1"]
    p["대학수"] = p["대학수_L1"]
    for g in ["다이나믹", "정체", "상권축소"]:
        p[f"chg_{g}"] = (p["상권변화지표_L1"] == g).astype(int)
    p["log_offset"] = np.log(p["독서실_점포수_L1"])
    p["동"] = p[DONG].astype("category").cat.codes
    p["분기FE"] = p[Q].astype(str)
    return p.dropna(subset=XN + ["독서실_폐업수"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. 다중공선성 점검 (VIF)
# ---------------------------------------------------------------------------
def vif_table(p: pd.DataFrame) -> pd.DataFrame:
    X = sm.add_constant(p[XN])
    rows = [{"변수": v, "표시": LABEL[v], "VIF": variance_inflation_factor(X.values, i + 1)}
            for i, v in enumerate(XN)]
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


# ---------------------------------------------------------------------------
# 3. 모형 추정
# ---------------------------------------------------------------------------
def cluster_kw(df: pd.DataFrame) -> dict:
    return {"cov_type": "cluster", "cov_kwds": {"groups": df["동"].values}}


def fit_nb(p: pd.DataFrame, x: list[str], fe: str = "C(분기FE) + C(자치구_코드)"):
    f = f"독서실_폐업수 ~ {' + '.join(x)} + {fe}"
    return smf.negativebinomial(f, data=p, offset=p["log_offset"]).fit(
        method="bfgs", maxiter=2000, disp=False, **cluster_kw(p))


def fit_poisson_fe(p: pd.DataFrame, x: list[str]):
    # 기간 내 폐업 합계가 0인 동은 동 FE가 -∞로 발산하므로 제외
    keep = p.groupby("동")["독서실_폐업수"].transform("sum") > 0
    d = p[keep].reset_index(drop=True)
    f = f"독서실_폐업수 ~ {' + '.join(x)} + C(분기FE) + C(동)"
    m = smf.glm(f, data=d, family=sm.families.Poisson(), offset=d["log_offset"]).fit(**cluster_kw(d))
    return m, d


def fit_logit(p: pd.DataFrame, x: list[str]):
    f = f"폐업발생 ~ {' + '.join(x)} + C(분기FE) + C(자치구_코드)"
    return smf.logit(f, data=p).fit(method="bfgs", maxiter=3000, disp=False, **cluster_kw(p))


def fit_clogit(p: pd.DataFrame, x: list[str]):
    keep = p.groupby("동")["폐업발생"].transform(lambda s: 0 < s.mean() < 1)   # 결과가 변하는 동만
    d = p[keep].reset_index(drop=True)
    qd = pd.get_dummies(d["분기FE"], prefix="q", drop_first=True, dtype=float)
    X = pd.concat([d[x].astype(float), qd], axis=1)
    m = ConditionalLogit(d["폐업발생"].values, X, groups=d["동"].values).fit(disp=False)
    return m, d


def coef_rows(m, model: str, n: int, n_dong: int, exp_name: str) -> list[dict]:
    ci = m.conf_int()
    rows = []
    for v in XN:
        if v not in m.params.index:
            continue
        b = m.params[v]
        rows.append({"모형": model, "변수": v, "표시": LABEL[v], "가설": HYP[v], "계수": b,
                     exp_name: np.exp(b), "95CI_하한": np.exp(ci.loc[v, 0]), "95CI_상한": np.exp(ci.loc[v, 1]),
                     "SE": m.bse[v], "p": m.pvalues[v], "n": n, "동_수": n_dong})
    return rows


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.1 else ""


# ---------------------------------------------------------------------------
# 4. 그래프: 모형별 효과 비교 (포레스트 플롯)
# ---------------------------------------------------------------------------
def forest(coefs: pd.DataFrame) -> None:
    models = [("M1 음이항 (구·분기 FE)", "배수"), ("M2 포아송 (동·분기 FE)", "배수"), ("M3 로지스틱 (구·분기 FE)", "배수")]
    fig, ax = plt.subplots(figsize=(10, 8))
    n = len(XN)
    offsets = [-0.25, 0, 0.25]
    markers = ["o", "s", "D"]
    for (name, col), off, c, mk in zip(models, offsets, SERIES, markers):
        d = coefs[coefs["모형"] == name].set_index("변수").reindex(XN)
        d = d.where(np.isfinite(d["95CI_상한"]) & (d["95CI_상한"] < 50))   # 동 FE에 흡수된 변수(추정 불가) 제외
        y = np.arange(n)[::-1] + off
        ax.errorbar(d[col], y, xerr=[d[col] - d["95CI_하한"], d["95CI_상한"] - d[col]],
                    fmt=mk, color=c, markersize=7, capsize=3, linewidth=1.6, label=name)
    ax.axvline(1, color=TEXT2, linewidth=1)
    ax.set_xscale("log")
    ticks = [0.5, 0.67, 0.8, 1, 1.25, 1.5, 2]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([f"{LABEL[v]}  [{HYP[v]}]" for v in XN], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("효과 배수 (IRR: 점포당 폐업률 배수 / OR: 폐업 발생 오즈 배수) — 1보다 크면 폐업 위험 증가")
    ax.set_title("요인별 폐업 위험 효과와 95% 신뢰구간 (동별 군집 표준오차)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_모형_계수_비교.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    p = prepare()
    n_all, d_all = len(p), p["동"].nunique()
    print(f"분석 대상: {n_all:,}행, {d_all}개 동\n")

    # VIF
    vif = vif_table(p)
    vif.to_csv(TAB / "t13_VIF.csv", index=False, encoding="utf-8-sig")
    print("[VIF] (10 이상이면 심각, 5 이상이면 주의)")
    for _, r in vif.iterrows():
        print(f"  {r['표시']:<28} {r['VIF']:6.2f}")

    coefs, fit = [], []

    print("\n[M1] 음이항 회귀 + 분기·자치구 FE ...")
    m1 = fit_nb(p, XN)
    alpha = m1.params["alpha"]
    coefs += coef_rows(m1, "M1 음이항 (구·분기 FE)", n_all, d_all, "배수")
    fit.append({"모형": "M1", "n": n_all, "동_수": d_all, "로그우도": m1.llf, "AIC": m1.aic,
                "과대산포_alpha": alpha, "수렴": m1.mle_retvals.get("converged")})

    print("[M2] 포아송 + 동·분기 FE ...")
    m2, d2 = fit_poisson_fe(p, XN)
    coefs += coef_rows(m2, "M2 포아송 (동·분기 FE)", len(d2), d2["동"].nunique(), "배수")
    fit.append({"모형": "M2", "n": len(d2), "동_수": d2["동"].nunique(), "로그우도": m2.llf, "AIC": m2.aic,
                "수렴": m2.converged})

    print("[M3] 로지스틱 + 분기·자치구 FE ...")
    m3 = fit_logit(p, XN)
    coefs += coef_rows(m3, "M3 로지스틱 (구·분기 FE)", n_all, d_all, "배수")
    fit.append({"모형": "M3", "n": n_all, "동_수": d_all, "로그우도": m3.llf, "AIC": m3.aic,
                "McFadden_R2": m3.prsquared, "수렴": m3.mle_retvals.get("converged")})

    print("[강건성] M1 — 2021년 제외 / 직장인구 추가 ...")
    p_no21 = p[p["연도"] > 2021].reset_index(drop=True)
    r1 = fit_nb(p_no21, XN)
    coefs += coef_rows(r1, "R1 음이항 (2021 제외)", len(p_no21), p_no21["동"].nunique(), "배수")
    p_w = p.dropna(subset=["lg_직장"]).reset_index(drop=True)
    r2 = fit_nb(p_w, XN + ["lg_직장"])
    coefs += coef_rows(r2, "R2 음이항 (+직장인구)", len(p_w), p_w["동"].nunique(), "배수")

    coefs = pd.DataFrame(coefs)
    coefs.to_csv(TAB / "t14_모형_계수.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fit).to_csv(TAB / "t15_모형_적합도.csv", index=False, encoding="utf-8-sig")

    # 요약 표: 모형(열) × 변수(행), 배수 + 유의성 별표
    wide = coefs.assign(값=coefs.apply(lambda r: f"{r['배수']:.3f}{stars(r['p'])}", axis=1)).pivot(
        index="변수", columns="모형", values="값").reindex(XN)
    wide.index = [LABEL[v] for v in XN]
    wide.to_csv(TAB / "t16_모형_요약표.csv", encoding="utf-8-sig")
    forest(coefs)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 100)
    print("효과 배수 (IRR / OR)   *** p<0.001  ** p<0.01  * p<0.05  † p<0.1   (동별 군집 표준오차, M4 제외)")
    print("=" * 100)
    print(wide.to_string())
    print(f"\nM1 과대산포 alpha = {alpha:.3f} (0보다 크면 포아송보다 음이항이 적합)")
    print(f"M3 McFadden R² = {m3.prsquared:.3f}")
    print("\n저장:", TAB / "t14_모형_계수.csv", "/", FIG / "fig10_모형_계수_비교.png")


if __name__ == "__main__":
    main()
