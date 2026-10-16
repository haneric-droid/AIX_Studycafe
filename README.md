# AIX_Studycafe — 스터디카페(독서실) 폐업률 영향요인 분석

Team456 (AI+X 공학융합프로젝트) · 서울시 공공데이터 기반

- **목표·계획**: [작업/분석목표_및_계획.md](작업/분석목표_및_계획.md)
- **분석 단위**: 서울 행정동 × 분기 (2021Q2~2026Q2), Y = 독서실·스터디카페 폐업률 (서울시 상권분석서비스)

## 폴더
| 경로 | 내용 |
|---|---|
| `보고서/` | 3~6주차 보고서 |
| `데이터/` | 초기 수집 데이터, 데이터 링크 목록 |
| `작업/src/` | 분석 코드 (01 수집 → 02 전처리 → 03 EDA → 04 가설검정 → 05 회귀모형) |
| `작업/data/processed/` | 행정동×분기 패널 데이터, 변수설명서 |
| `작업/outputs/` | 그래프, 결과 표, EDA·가설검정 결과 요약 |

## 재현 방법
```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn requests openpyxl
cd 작업
python src/01_download.py   # 서울 Open API 인증키를 작업/api_key.txt 에 넣어야 점포-행정동 수집 가능
python src/02_preprocess.py
python src/03_eda.py
python src/04_hypothesis.py
python src/05_model.py
```
원본 데이터(`작업/data/raw/`)와 인증키는 저장소에 포함하지 않았습니다.
