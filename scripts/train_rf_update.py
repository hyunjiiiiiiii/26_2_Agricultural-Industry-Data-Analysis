import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# =========================
# 설정
# =========================
DATA_PATH = "rf_stats_data_Y.xlsx"
GROUP_COL = "area"
TARGET_COL = "Y"

N_REPEATS = 30            # ← 반복 횟수 (20~50 추천)
TEST_SIZE_TARGET = 0.30
MIN_PER_CLASS = 20        # 너무 크면 split 못 찾음
RANDOM_STATE = 42

# =========================
# 데이터 로드
# =========================
df = pd.read_excel(DATA_PATH)
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
df[TARGET_COL] = df[TARGET_COL].astype(str)
df = df[df[TARGET_COL] != "unknown"].copy()

# X / y
y_str = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL]).copy()

for col in ["id"]:
    if col in X.columns:
        X = X.drop(columns=[col])

le = LabelEncoder()
y = le.fit_transform(y_str)
classes = le.classes_
n_classes = len(classes)

groups = df[GROUP_COL].astype(str).values
unique_groups = np.unique(groups)

# 그룹별 인덱스
group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
group_sizes = {g: len(group_to_idx[g]) for g in unique_groups}
N = len(df)
target_test_n = int(N * TEST_SIZE_TARGET)

rng = np.random.default_rng(RANDOM_STATE)

# =========================
# helper 함수
# =========================
def label_counts(indices):
    return np.bincount(y[indices], minlength=n_classes)

def valid_test(indices):
    return np.all(label_counts(indices) >= MIN_PER_CLASS)

# =========================
# 반복 평가
# =========================
accs, f1s = [], []
cm_sum = np.zeros((n_classes, n_classes), dtype=int)

successful = 0
attempts = 0
MAX_TEST_GROUPS = max(1, len(unique_groups) - 1)
while successful < N_REPEATS and attempts < N_REPEATS * 500:
    attempts += 1

    # 랜덤 섞기
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)

    chosen, total = [], 0
    for g in shuffled:
        # ✅ test 그룹이 전체 그룹을 먹지 않게 제한
        if len(chosen) >= MAX_TEST_GROUPS:
            break

        if total < target_test_n:
            chosen.append(g)
            total += group_sizes[g]
        else:
            break

    # ✅ chosen이 비었거나(=test 0) 혹은 전체 그룹이면 skip
    if len(chosen) == 0 or len(chosen) == len(unique_groups):
        continue

    test_idx = np.concatenate([group_to_idx[g] for g in chosen])
    train_idx = np.setdiff1d(np.arange(N), test_idx)

    # ✅ train/test 0 방지
    if len(train_idx) == 0 or len(test_idx) == 0:
        continue

    train_idx = np.setdiff1d(np.arange(N), test_idx)

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    # area는 그룹이므로 feature에서 제거
    X_train = X_train.drop(columns=[GROUP_COL], errors="ignore")
    X_test  = X_test.drop(columns=[GROUP_COL], errors="ignore")

    # RF
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=rng.integers(0, 1e9),
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    accs.append(accuracy_score(y_test, pred))
    f1s.append(f1_score(y_test, pred, average="macro", zero_division=0))
    cm_sum += confusion_matrix(y_test, pred, labels=np.arange(n_classes))

    successful += 1

# =========================
# 결과 요약
# =========================
print("\n==============================")
print(f"Successful splits: {successful}/{attempts}")
print("Classes:", list(classes))
print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print("\nConfusion matrix (sum over repeats):")
print(cm_sum)
