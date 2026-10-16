"""
02_preprocess.py — 행정동 × 분기 패널 데이터 생성

입력 : 작업/data/raw/seoul/  (01_download.py 결과)
출력 : 작업/data/processed/panel_dong_quarter.csv   (분석용 패널)
       작업/data/processed/panel_dong_quarter_all.csv (독서실이 없는 동-분기까지 포함한 전체 격자)

패널 설계
  - 행 = 행정동(425개) × 분기(2021Q2~2026Q2). 2021Q1은 직전 분기 값이 없어 제외.
  - Y(종속변수) = 같은 분기(t)의 독서실 폐업 지표
  - X(설명변수) = 직전 분기(t-1) 값 → 컬럼명 끝에 `_L1`
    (원인이 결과보다 먼저 일어나도록 시간 순서를 맞추기 위함)
  - 분석 대상 = t 또는 t-1 시점에 독서실이 1개 이상 있는 동-분기
  - 임대료 데이터 대신 아파트 평균 시가를 고정비(지가 수준)의 대리변수로 사용

사용법 (작업 폴더에서)
  python src/02_preprocess.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]          # 작업/
RAW = ROOT / "data" / "raw" / "seoul"
OUT = ROOT / "data" / "processed"

Q, DONG = "기준_년분기_코드", "행정동_코드"
KEY = [Q, DONG]
ACADEMIES = ["일반교습학원", "외국어학원", "예술학원", "컴퓨터학원"]


def read(name: str, cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{name}.csv", dtype={DONG: str, "자치구_코드": str})
    df[Q] = df[Q].astype(int)
    return df if cols is None else df[KEY + cols]


def next_quarter(q: int) -> int:
    """20214 → 20221, 20213 → 20214"""
    y, n = divmod(q, 10)
    return (y + 1) * 10 + 1 if n == 4 else q + 1


# ---------------------------------------------------------------------------
# 1. 뼈대: 전체 행정동 × 전체 분기 격자
# ---------------------------------------------------------------------------
def build_grid() -> pd.DataFrame:
    pop = read("상권분석_상주인구_행정동")
    dongs = pop[[DONG, "행정동_코드_명"]].drop_duplicates(DONG)
    quarters = sorted(pop[Q].unique())
    grid = dongs.merge(pd.DataFrame({Q: quarters}), how="cross")
    # 자치구: 행정동 코드 앞 5자리 = 자치구 코드
    grid["자치구_코드"] = grid[DONG].str[:5]
    gu = read("상권분석_상주인구_자치구")[["자치구_코드", "자치구_코드_명"]].drop_duplicates()
    grid = grid.merge(gu.rename(columns={"자치구_코드_명": "자치구명"}), on="자치구_코드", how="left")
    return grid.rename(columns={"행정동_코드_명": "행정동명"})


# ---------------------------------------------------------------------------
# 2. 점포 데이터: 독서실(Y + 경쟁 변수), 학원(학원가 변수)
# ---------------------------------------------------------------------------
def store_features() -> pd.DataFrame:
    dok = read("상권분석_점포_행정동_독서실").rename(columns={
        "전체_점포_수": "독서실_점포수",
        "프랜차이즈_점포_수": "독서실_프랜차이즈수",
        "개업_점포_수": "독서실_개업수",
        "개업_율": "독서실_개업률",
        "폐업_점포_수": "독서실_폐업수",
        "폐업_률": "독서실_폐업률",
    })
    dok = dok[KEY + ["독서실_점포수", "독서실_프랜차이즈수", "독서실_개업수",
                     "독서실_개업률", "독서실_폐업수", "독서실_폐업률"]]

    allst = pd.read_csv(RAW / "상권분석_점포_행정동.csv", dtype={DONG: str},
                        usecols=KEY + ["서비스_업종_코드_명", "전체_점포_수"])
    allst = allst[allst["서비스_업종_코드_명"].isin(ACADEMIES)]
    aca = allst.pivot_table(index=KEY, columns="서비스_업종_코드_명", values="전체_점포_수",
                            aggfunc="sum", fill_value=0).reset_index()
    aca.columns.name = None
    aca["학원_합계수"] = aca[ACADEMIES].sum(axis=1)
    aca = aca.rename(columns={"일반교습학원": "학원_일반교습수"})[KEY + ["학원_일반교습수", "학원_합계수"]]
    return dok.merge(aca, on=KEY, how="outer")


# ---------------------------------------------------------------------------
# 3. 수요·입지·고정비 변수
# ---------------------------------------------------------------------------
def area_features() -> pd.DataFrame:
    pop = read("상권분석_상주인구_행정동")
    pop["상주_10_20대"] = pop["연령대_10_상주인구_수"] + pop["연령대_20_상주인구_수"]
    pop = pop[KEY + ["총_상주인구_수", "연령대_10_상주인구_수", "상주_10_20대", "총_가구_수"]].rename(
        columns={"총_상주인구_수": "상주_총", "연령대_10_상주인구_수": "상주_10대", "총_가구_수": "가구_총"})

    work = read("상권분석_직장인구_행정동", ["총_직장_인구_수"]).rename(columns={"총_직장_인구_수": "직장_총"})

    flow = read("상권분석_길단위인구_행정동")
    flow["유동_10_20대"] = flow["연령대_10_유동인구_수"] + flow["연령대_20_유동인구_수"]
    flow = flow[KEY + ["총_유동인구_수", "유동_10_20대", "시간대_21_24_유동인구_수"]].rename(
        columns={"총_유동인구_수": "유동_총", "시간대_21_24_유동인구_수": "유동_야간21_24"})

    fac = read("상권분석_집객시설_행정동")
    fac_cols = ["초등학교_수", "중학교_수", "고등학교_수", "대학교_수", "지하철_역_수", "버스_정거장_수"]
    fac[fac_cols] = fac[fac_cols].fillna(0)            # 결측 = 해당 시설 없음
    fac["학교_초중고수"] = fac[["초등학교_수", "중학교_수", "고등학교_수"]].sum(axis=1)
    fac = fac[KEY + ["학교_초중고수", "고등학교_수", "대학교_수", "지하철_역_수", "버스_정거장_수"]].rename(
        columns={"고등학교_수": "고교수", "대학교_수": "대학수", "지하철_역_수": "지하철역수",
                 "버스_정거장_수": "버스정거장수"})

    spend = read("상권분석_소비_행정동", ["지출_총금액", "교육_지출_총금액"]).rename(
        columns={"지출_총금액": "지출_총액", "교육_지출_총금액": "교육지출_총액"})
    spend["교육지출_비중"] = spend["교육지출_총액"] / spend["지출_총액"].replace(0, np.nan)

    apt = read("상권분석_아파트_행정동")
    area_cols = [c for c in apt.columns if c.startswith("아파트_면적_") and c.endswith("세대_수")]
    apt["아파트_세대수"] = apt[area_cols].fillna(0).sum(axis=1)
    apt["아파트_평균시가_억"] = apt["아파트_평균_시가"] / 1e8
    apt = apt[KEY + ["아파트_단지_수", "아파트_세대수", "아파트_평균시가_억"]].rename(
        columns={"아파트_단지_수": "아파트_단지수"})

    chg = read("상권분석_상권변화지표_행정동",
               ["상권_변화_지표_명", "운영_영업_개월_평균", "폐업_영업_개월_평균"]).rename(
        columns={"상권_변화_지표_명": "상권변화지표", "운영_영업_개월_평균": "상권_운영개월평균",
                 "폐업_영업_개월_평균": "상권_폐업개월평균"})

    out = pop
    for df in [work, flow, fac, spend, apt, chg]:
        out = out.merge(df, on=KEY, how="left")
    return out


# ---------------------------------------------------------------------------
# 4. 결합 → 결측 처리 → 파생변수 → 시차(t-1)
# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_grid().merge(store_features(), on=KEY, how="left").merge(area_features(), on=KEY, how="left")
    df = df.sort_values([DONG, Q]).reset_index(drop=True)

    # 점포 행이 없는 동-분기 = 해당 업종 점포 0개
    zero_cols = ["독서실_점포수", "독서실_프랜차이즈수", "독서실_개업수", "독서실_개업률",
                 "독서실_폐업수", "독서실_폐업률", "학원_일반교습수", "학원_합계수", "아파트_단지수", "아파트_세대수"]
    df[zero_cols] = df[zero_cols].fillna(0)

    # 아파트 시가 결측 → 같은 동의 인접 분기 값 → 그래도 없으면 같은 분기 자치구 중앙값
    df["아파트시가_보간"] = df["아파트_평균시가_억"].isna().astype(int)
    df["아파트_평균시가_억"] = df.groupby(DONG)["아파트_평균시가_억"].transform(lambda s: s.ffill().bfill())
    df["아파트_평균시가_억"] = df["아파트_평균시가_억"].fillna(
        df.groupby(["자치구_코드", Q])["아파트_평균시가_억"].transform("median"))

    # 파생변수 (경쟁 강도·비율)
    df["독서실_프랜차이즈비율"] = df["독서실_프랜차이즈수"] / df["독서실_점포수"].replace(0, np.nan)
    df["독서실_per_1천명_10_20대"] = df["독서실_점포수"] / (df["상주_10_20대"] / 1000).replace(0, np.nan)
    df["학원_per_독서실"] = df["학원_일반교습수"] / df["독서실_점포수"].replace(0, np.nan)

    # 직전 분기(t-1) 값 붙이기: 분기 코드를 한 칸 당겨서 병합
    x_cols = [
        # 경쟁
        "독서실_점포수", "독서실_프랜차이즈비율", "독서실_개업수", "독서실_per_1천명_10_20대",
        # 학원가
        "학원_일반교습수", "학원_합계수", "학원_per_독서실",
        # 수요
        "상주_총", "상주_10대", "상주_10_20대", "가구_총", "직장_총", "유동_총", "유동_10_20대", "유동_야간21_24",
        # 시설
        "학교_초중고수", "고교수", "대학수", "지하철역수", "버스정거장수",
        # 소득·소비
        "지출_총액", "교육지출_총액", "교육지출_비중",
        # 고정비 대리변수
        "아파트_단지수", "아파트_세대수", "아파트_평균시가_억",
        # 상권 상태
        "상권변화지표", "상권_운영개월평균", "상권_폐업개월평균",
    ]
    lag = df[KEY + x_cols].copy()
    lag[Q] = lag[Q].map(next_quarter)                   # t-1 값을 t 행에 붙임
    lag = lag.rename(columns={c: f"{c}_L1" for c in x_cols})
    panel = df[KEY + ["자치구_코드", "자치구명", "행정동명",
                      "독서실_점포수", "독서실_개업수", "독서실_폐업수", "독서실_폐업률", "아파트시가_보간"]]
    panel = panel.rename(columns={"독서실_점포수": "독서실_점포수_t", "독서실_개업수": "독서실_개업수_t"})
    panel = panel.merge(lag, on=KEY, how="inner")       # 2021Q1은 t-1이 없어 자동 제외

    # Y 파생
    panel["폐업발생"] = (panel["독서실_폐업수"] > 0).astype(int)
    panel["분기"] = panel[Q] % 10
    panel["연도"] = panel[Q] // 10

    # 과밀 더미: 같은 분기, 독서실이 있는 동 중에서 경쟁 강도 상위 25% = 1
    has_store = panel[panel["독서실_점포수_L1"] > 0]
    for src, name in [("독서실_점포수_L1", "과밀_점포수"), ("독서실_per_1천명_10_20대_L1", "과밀_인구대비")]:
        cut = panel[Q].map(has_store.groupby(Q)[src].quantile(0.75))
        panel[name] = (panel[src] > cut).astype(int)

    # 분석 대상: t 또는 t-1에 독서실이 있는 동-분기
    panel["분석대상"] = ((panel["독서실_점포수_t"] > 0) | (panel["독서실_점포수_L1"] > 0)).astype(int)

    panel = panel.sort_values([DONG, Q]).reset_index(drop=True)
    panel.to_csv(OUT / "panel_dong_quarter_all.csv", index=False, encoding="utf-8-sig")
    ana = panel[panel["분석대상"] == 1].reset_index(drop=True)
    ana.to_csv(OUT / "panel_dong_quarter.csv", index=False, encoding="utf-8-sig")

    # ---------------- 품질 리포트 ----------------
    print(f"전체 격자: {len(panel):,}행 ({panel[DONG].nunique()}개 동 × {panel[Q].nunique()}개 분기)")
    print(f"분석 대상: {len(ana):,}행 ({ana[DONG].nunique()}개 동), 컬럼 {ana.shape[1]}개")
    print(f"폐업 발생 비율: {ana['폐업발생'].mean():.1%}, 평균 폐업 수 {ana['독서실_폐업수'].mean():.3f}, "
          f"평균 폐업률 {ana['독서실_폐업률'].mean():.2f}%")
    print(f"과밀(점포수) 비율: {ana['과밀_점포수'].mean():.1%} / 과밀(인구대비) 비율: {ana['과밀_인구대비'].mean():.1%}")
    print(f"아파트 시가 보간 행: {ana['아파트시가_보간'].sum()}")
    na = ana.isna().sum()
    print("결측 있는 컬럼:", na[na > 0].to_dict() or "없음")
    print("저장:", OUT / "panel_dong_quarter.csv")


if __name__ == "__main__":
    main()
