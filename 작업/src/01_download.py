"""
01_download.py — 스터디카페(독서실) 폐업률 분석용 원본 데이터 자동 수집

수집 대상
  1) sheet : 서울 열린데이터광장 상권분석서비스 (인구·시설·소비·아파트·상권변화·점포-자치구)
             → 시트 CSV 내려받기 (인증키 불필요)
  2) api   : 서울시 상권분석서비스(점포-행정동, OA-22172) → Open API (인증키 필요)
             시트 내려받기는 약 21만 행에서 잘리므로 API로 분기별 전체를 받는다.
  3) nts   : 국세청_사업자현황_100대 생활업종 (월별, 자치구) → 공공데이터포털 (교차검증용)

인증키 설정 (둘 중 하나)
  - 환경변수  SEOUL_API_KEY
  - 파일      작업/api_key.txt  (첫 줄에 키만 적기)

사용법 (작업 폴더에서)
  python src/01_download.py                 # 전부 수집
  python src/01_download.py --only api      # 점포-행정동만
  python src/01_download.py --only sheet nts --force   # 기존 파일 덮어쓰기

결과는 작업/data/raw/ 아래에 UTF-8(BOM) CSV로 저장된다.
"""

import argparse
import io
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# 윈도우 콘솔(cp949)에서 한글/특수문자 출력 오류 방지
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]          # 작업/
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0"}

# ---------------------------------------------------------------------------
# 1) 서울 열린데이터광장 — 시트 CSV 내려받기
# ---------------------------------------------------------------------------
SEOUL_SHEETS = {
    # infId : 저장 파일명
    "OA-22173": "상권분석_점포_자치구",
    "OA-22182": "상권분석_상주인구_자치구",
    "OA-22183": "상권분석_상주인구_행정동",
    "OA-22184": "상권분석_직장인구_행정동",
    "OA-22185": "상권분석_직장인구_자치구",
    "OA-22179": "상권분석_길단위인구_자치구",
    "OA-22178": "상권분석_길단위인구_행정동",
    "OA-15575": "상권분석_상권변화지표_행정동",
    "OA-15567": "상권분석_상권변화지표_자치구",
    "OA-22167": "상권분석_소비_자치구",
    "OA-22166": "상권분석_소비_행정동",
    "OA-22170": "상권분석_집객시설_자치구",
    "OA-22169": "상권분석_집객시설_행정동",
    "OA-22164": "상권분석_아파트_자치구",
    "OA-22163": "상권분석_아파트_행정동",
}
SHEET_URL = "https://datafile.seoul.go.kr/bigfile/iot/sheet/csv/download.do"
SHEET_ROW_LIMIT_WARN = 200_000   # 이 이상이면 시트 다운로드가 잘렸을 가능성


def download_seoul_sheets(force: bool) -> None:
    out_dir = RAW / "seoul"
    out_dir.mkdir(parents=True, exist_ok=True)
    for inf_id, name in SEOUL_SHEETS.items():
        path = out_dir / f"{name}.csv"
        if path.exists() and not force:
            print(f"[sheet] 건너뜀(이미 있음): {path.name}")
            continue
        form = {
            "srvType": "S", "infId": inf_id, "serviceKind": "1", "pageNo": "1",
            "ssUserId": "SAMPLE_VIEW", "strWhere": "", "strOrderby": "STDR_YYQU_CD DESC",
        }
        r = requests.post(SHEET_URL, data=form, headers={**UA, "Referer": "https://data.seoul.go.kr/"},
                          timeout=300)
        r.raise_for_status()
        if "attachment" not in r.headers.get("Content-Disposition", ""):
            print(f"[sheet] 실패: {inf_id} (파일이 아닌 응답)")
            continue
        df = pd.read_csv(io.BytesIO(r.content), encoding="cp949")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        q = df.iloc[:, 0]
        print(f"[sheet] {inf_id} {name}: {len(df):,}행, 분기 {q.min()}~{q.max()}")
        if len(df) >= SHEET_ROW_LIMIT_WARN:
            print(f"   ⚠ {len(df):,}행 — 시트 다운로드 한도에 걸렸을 수 있음. API로 받는 것을 권장.")
        time.sleep(1)


# ---------------------------------------------------------------------------
# 2) 서울 Open API — 점포-행정동 (VwsmAdstrdStorW)
# ---------------------------------------------------------------------------
API_SERVICE = "VwsmAdstrdStorW"
API_PAGE = 1000                     # API 1회 최대 1000건
API_FIRST_QUARTER = (2021, 1)       # 서울 상권분석서비스는 2021Q1부터 제공
API_COLUMNS = {                     # API 영문 컬럼 → 시트 CSV와 같은 한글 컬럼명
    "STDR_YYQU_CD": "기준_년분기_코드",
    "ADSTRD_CD": "행정동_코드",
    "ADSTRD_CD_NM": "행정동_코드_명",
    "SVC_INDUTY_CD": "서비스_업종_코드",
    "SVC_INDUTY_CD_NM": "서비스_업종_코드_명",
    "SIMILR_INDUTY_STOR_CO": "전체_점포_수",
    "STOR_CO": "일반_점포_수",
    "FRC_STOR_CO": "프랜차이즈_점포_수",
    "OPBIZ_RT": "개업_율",
    "OPBIZ_STOR_CO": "개업_점포_수",
    "CLSBIZ_RT": "폐업_률",
    "CLSBIZ_STOR_CO": "폐업_점포_수",
}


def load_api_key() -> str | None:
    key = os.environ.get("SEOUL_API_KEY", "").strip()
    if key:
        return key
    key_file = ROOT / "api_key.txt"
    if key_file.exists():
        lines = key_file.read_text(encoding="utf-8").strip().splitlines()
        if lines:
            return lines[0].strip()
    return None


def api_call(key: str, start: int, end: int, quarter: str, retries: int = 4) -> dict:
    """API 한 페이지 요청. 일시적 오류는 잠시 쉬었다가 재시도."""
    url = f"http://openapi.seoul.go.kr:8088/{key}/json/{API_SERVICE}/{start}/{end}/{quarter}"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            # 정상 응답이면 {서비스명: {...}}, 오류면 {"RESULT": {...}}
            body = data.get(API_SERVICE, data)
            code = body.get("RESULT", {}).get("CODE", "")
            if code in ("INFO-000", "INFO-200"):
                return body
            if code in ("INFO-100", "ERROR-310", "ERROR-300", "ERROR-301"):
                raise SystemExit(f"[api] 요청 오류 {code}: {body['RESULT'].get('MESSAGE')}")
            print(f"   재시도 {attempt}/{retries} ({code})")
        except (requests.RequestException, ValueError) as e:
            print(f"   재시도 {attempt}/{retries} ({e.__class__.__name__})")
        time.sleep(2 * attempt)
    raise RuntimeError(f"[api] {quarter} {start}-{end} 요청 실패")


def iter_quarters():
    """2021Q1부터 데이터가 없는 분기가 나올 때까지 분기 코드(예: '20211')를 생성."""
    y, q = API_FIRST_QUARTER
    while True:
        yield f"{y}{q}"
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)


def download_api_store(force: bool) -> None:
    key = load_api_key()
    if not key:
        print("[api] 인증키가 없습니다. 환경변수 SEOUL_API_KEY 또는 작업/api_key.txt 를 설정하세요. (건너뜀)")
        return
    q_dir = RAW / "seoul_api" / API_SERVICE
    q_dir.mkdir(parents=True, exist_ok=True)

    for quarter in iter_quarters():
        path = q_dir / f"{quarter}.csv"
        if path.exists() and not force:
            print(f"[api] 건너뜀(이미 있음): {quarter}")
            continue
        first = api_call(key, 1, API_PAGE, quarter)
        total = int(first.get("list_total_count", 0))
        if total == 0:                      # 아직 공개되지 않은 분기 → 종료
            print(f"[api] {quarter}: 데이터 없음 → 수집 종료")
            break
        rows = list(first.get("row", []))
        for start in range(API_PAGE + 1, total + 1, API_PAGE):
            body = api_call(key, start, min(start + API_PAGE - 1, total), quarter)
            rows.extend(body.get("row", []))
            time.sleep(0.2)                 # 서버 부담 줄이기
        if len(rows) != total:
            raise RuntimeError(f"[api] {quarter}: 받은 행 {len(rows)} ≠ 총 {total}")
        pd.DataFrame(rows).rename(columns=API_COLUMNS).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[api] {quarter}: {total:,}행 저장")

    # 분기별 파일을 하나로 합치기 (+ 독서실만 뽑은 파일)
    files = sorted(q_dir.glob("*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f, dtype={"행정동_코드": str}) for f in files], ignore_index=True)
    df.to_csv(RAW / "seoul" / "상권분석_점포_행정동.csv", index=False, encoding="utf-8-sig")
    dok = df[df["서비스_업종_코드_명"] == "독서실"]
    dok.to_csv(RAW / "seoul" / "상권분석_점포_행정동_독서실.csv", index=False, encoding="utf-8-sig")
    print(f"[api] 합본 {len(df):,}행 / 독서실 {len(dok):,}행 "
          f"({dok['행정동_코드'].nunique()}개 동, {dok['기준_년분기_코드'].nunique()}개 분기)")


# ---------------------------------------------------------------------------
# 3) 공공데이터포털 — 국세청 100대 생활업종 (월별 이력 포함)
# ---------------------------------------------------------------------------
DG = "https://www.data.go.kr"
NTS_PK = "15061118"
NTS_FROM = "20201201"   # 2021Q1 직전부터 (분석 기간과 맞춤)


def dg_file_info(pk: str, detail_pk: str) -> tuple[str, str]:
    """상세 PK → (첨부파일 ID, 원본 파일명)."""
    r = requests.get(f"{DG}/tcs/dss/selectFileDataDownload.do", headers=UA, timeout=60,
                     params={"publicDataPk": pk, "publicDataDetailPk": detail_pk, "fileDetailSn": "1"})
    r.raise_for_status()
    vo = r.json()["fileDataRegistVO"]
    return vo["atchFileId"], vo["orginlFileNm"]


def download_nts(force: bool) -> None:
    out_dir = RAW / "nts_100"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 최신 파일의 상세 PK
    page = requests.get(f"{DG}/data/{NTS_PK}/fileData.do", headers=UA, timeout=60).text
    cur = re.search(r'name="publicDataDetailPk" value="(uddi:[^"]+)"', page).group(1)
    # 과거 이력 목록 (파일명, 상세 PK)
    hist = requests.get(f"{DG}/tcs/dss/selectHistAndCsvData.do", headers=UA, timeout=60,
                        params={"publicDataPk": NTS_PK, "publicDataDetailPk": cur}).text
    items = re.findall(r'data-public-pk="(uddi:[^"]+)"[^>]*>\s*([^<]+?)\s*</a>', hist)
    targets = [(cur, None)] + [(p, n) for p, n in items]

    for detail_pk, name in targets:
        m = re.search(r"(\d{8})", name or "")
        if m and m.group(1) < NTS_FROM:
            continue
        atch_id, fname = dg_file_info(NTS_PK, detail_pk)
        path = out_dir / fname
        if path.exists() and not force:
            print(f"[nts] 건너뜀(이미 있음): {fname}")
            continue
        r = requests.get(f"{DG}/cmm/cmm/fileDownload.do", headers=UA, timeout=120,
                         params={"atchFileId": atch_id, "fileDetailSn": "1", "insertDataPrcus": "N"})
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"[nts] {fname} ({len(r.content) / 1e6:.1f}MB)")
        time.sleep(0.5)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="스터디카페 폐업률 분석 원본 데이터 수집")
    ap.add_argument("--only", nargs="+", choices=["sheet", "api", "nts"],
                    default=["sheet", "api", "nts"], help="수집할 대상 (기본: 전부)")
    ap.add_argument("--force", action="store_true", help="이미 받은 파일도 다시 받기")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "seoul").mkdir(exist_ok=True)
    if "sheet" in args.only:
        download_seoul_sheets(args.force)
    if "api" in args.only:
        download_api_store(args.force)
    if "nts" in args.only:
        download_nts(args.force)
    print("완료:", RAW)


if __name__ == "__main__":
    main()
