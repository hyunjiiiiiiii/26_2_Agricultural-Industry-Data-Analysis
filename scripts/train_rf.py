import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# 1) 데이터 로드 (헤더 문제 해결된 상태 가정)
df = pd.read_excel("rf_stats_data_env.xlsx")

# 2) Y 확인
assert "Y" in df.columns

# 3) X / y 분리
y_str = df["Y"]
X = df.drop(columns=["Y"])

# 4) id 있으면 제거
for col in ["id", "Unnamed: 0"]:
    if col in X.columns:
        X = X.drop(columns=[col])

# 5) Label encoding (Y만)
le = LabelEncoder()
y = le.fit_transform(y_str)

print("label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 6) train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# 7) Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# 8) 평가
pred = rf.predict(X_test)

print(classification_report(
    y_test,
    pred,
    labels=np.arange(len(le.classes_)),   # [0,1,2,3]
    target_names=le.classes_,
    zero_division=0
))
print("Confusion matrix:\n",
      confusion_matrix(
          y_test,
          pred,
          labels=np.arange(len(le.classes_))
      )
)


# ✅ (2) importance index는 학습에 사용한 컬럼으로!
importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nFeature importances (top 30):")
print(importances.head(30))

# =========================
# 9) 예측 기준 B / normal만 저장
# =========================

# 원본 df 복사
df_result = df.copy()

# 예측 결과 컬럼 추가
df_result["y_pred"] = np.nan
df_result.loc[X_test.index, "y_pred"] = pred

# label index 확인
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
inv_map = {v: k for k, v in label_map.items()}

b_label = label_map["B"]
normal_label = label_map["normal"]

# 예측 기준 B / normal만 선택
df_pred_bn = df_result[df_result["y_pred"].isin([b_label, normal_label])].copy()

# 사람이 읽기 좋은 라벨 추가
df_pred_bn["y_pred_str"] = df_pred_bn["y_pred"].map(inv_map)

# 저장
df_pred_bn.to_excel("rf_predicted_B_vs_normal.xlsx", index=False)

print("\n[Saved predicted B vs normal samples]")
print("file: rf_predicted_B_vs_normal.xlsx")
print(df_pred_bn["y_pred_str"].value_counts())

