import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import classification_report

from tabpfn import TabPFNClassifier

# 엑셀 읽기 (Y가 이미 만들어진 파일을 쓰면 편함)
df = pd.read_excel("stats_data_env.xlsx")
print(df.columns)

# 1) unknown 제거(있다면)
df = df[df["Y"] != "unknown"].copy()

# 2) X / y 분리
y_str = df["Y"]                         # 'A','B','C','normal'
X = df.drop(columns=["Y"])              # 나머지 전부가 feature 후보

# 3) id 제거 (모델에 넣지 않는 게 일반적으로 안전)
if "id" in X.columns:
    X = X.drop(columns=["id"])

# 4) 범주형 컬럼 지정
cat_cols = ["area", "grow"]
num_cols = [c for c in X.columns if c not in cat_cols]

# 5) 전처리: 결측 + 범주형 인코딩(Ordinal)
preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols),
    ],
    remainder="drop",
)

# 6) y 라벨도 모델용 숫자로 변환 (TabPFN은 문자열도 되는 경우가 있지만, 숫자가 안전)
le = LabelEncoder()
y = le.fit_transform(y_str)  # A/B/C/normal -> 0/1/2/3 (순서는 le.classes_ 확인)

# 7) train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8) 모델
model = Pipeline(steps=[
#    ("prep", preprocess),
    ("clf", TabPFNClassifier(device="cuda", n_estimators=16))  # GPU 없으면 device="cpu"
])

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(classification_report(y_test, pred, target_names=le.classes_))
