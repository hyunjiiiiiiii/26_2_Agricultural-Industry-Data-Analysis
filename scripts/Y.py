import re
import pandas as pd

# 1) 엑셀 로드
path = "stats_data_real_Y.xlsx"          # 너 파일명으로 바꿔
sheet = "stats_data"            # 시트명 바꿔도 됨
df = pd.read_excel(path, sheet_name=sheet)

# 2) id -> Y 라벨 생성 함수
pattern = re.compile(r"(?:^|_)(00|[abc]\d+)(?:_|$)", re.IGNORECASE)

def make_y_from_id(id_str: str) -> str:
    s = str(id_str)
    found = pattern.findall(s)

    found = [f.lower() for f in found]
    if any(f.startswith("c") for f in found): return "C"
    if any(f.startswith("b") for f in found): return "B"
    if any(f.startswith("a") for f in found): return "A"
    else: return "normal"

df["Y"] = df["id"].apply(make_y_from_id)

# (선택) unknown이 있는지 확인
print(df["Y"].value_counts(dropna=False))

# (선택) Y까지 포함해 새 파일로 저장 (엑셀에 'Y'열이 생김)
df.to_excel("stats_data_Y.xlsx", index=False)
