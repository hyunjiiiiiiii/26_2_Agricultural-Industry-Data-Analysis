import pandas as pd

# 1) 데이터 로드
df = pd.read_excel("rf_stats_data_Y.xlsx")

# 2) Y 컬럼 확인
assert "Y" in df.columns, "Y 컬럼이 없습니다."

# 3) B / normal만 필터링
df_bn = df[df["Y"].astype(str).isin(["B", "normal"])].copy()

# 4) 저장
out_path = "rf_stats_data_B_vs_normal.xlsx"
df_bn.to_excel(out_path, index=False)

# 5) 확인 출력
print(f"✅ saved: {out_path}")
print("label counts:")
print(df_bn["Y"].value_counts())
