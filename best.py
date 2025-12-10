import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVDpp
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================

train_path = 'train.csv'
test_path = 'test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 범주형 변수 String 변환
train_df['genres'] = train_df['genres'].astype(str)
test_df['genres'] = test_df['genres'].astype(str)
train_df['title'] = train_df['title'].astype(str)
test_df['title'] = test_df['title'].astype(str)

train_df = train_df.dropna()

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ==========================================
# 2. 모델 정의 및 학습 함수
# ==========================================

def run_svdpp(train_data, val_data=None, full_train=False):
    # Rating Scale 설정
    reader = Reader(rating_scale=(0.5, 5.0))
    
    if full_train:
        data = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        # n_factors 등 하이퍼파라미터는 시간과 성능에 따라 조절 가능
        model = SVDpp(n_factors=20, random_state=42, verbose=True) 
        model.fit(trainset)
        return model
    else:
        data = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        model = SVDpp(n_factors=20, random_state=42, verbose=False)
        model.fit(trainset)
        
        preds = []
        for _, row in val_data.iterrows():
            preds.append(model.predict(row['userId'], row['movieId']).est)
        return model, np.array(preds)

def run_catboost(train_data, val_data=None, full_train=False):
    features = ['userId', 'movieId', 'year', 'genres']
    cat_features = ['userId', 'movieId', 'genres']
    
    X = train_data[features]
    y = train_data['rating']
    
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        cat_features=cat_features,
        verbose=100,
        random_seed=42
    )
    
    if full_train:
        model.fit(X, y)
        return model
    else:
        X_val = val_data[features]
        y_val = val_data['rating']
        
        model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_val)
        return model, preds

# ==========================================
# 3. Validation Phase (검증)
# ==========================================
print("\n========== Validation Phase ==========")

X_train, X_val = train_test_split(train_df, test_size=0.2, random_state=42)

print("Training SVD++ on split data...")
_, svd_val_preds = run_svdpp(X_train, X_val, full_train=False)

print("Training CatBoost on split data...")
_, cat_val_preds = run_catboost(X_train, X_val, full_train=False)

# RMSE 확인 (반올림 전 성능)
rmse_svd = np.sqrt(mean_squared_error(X_val['rating'], svd_val_preds))
rmse_cat = np.sqrt(mean_squared_error(X_val['rating'], cat_val_preds))
print(f"SVD++ RMSE (Raw): {rmse_svd:.4f}")
print(f"CatBoost RMSE (Raw): {rmse_cat:.4f}")

# 최적 가중치 탐색
best_rmse = float('inf')
best_weight = 0.5
for w in np.linspace(0, 1, 101):
    ensemble_preds = (w * svd_val_preds) + ((1 - w) * cat_val_preds)
    current_rmse = np.sqrt(mean_squared_error(X_val['rating'], ensemble_preds))
    if current_rmse < best_rmse:
        best_rmse = current_rmse
        best_weight = w

print(f"Best Weight: {best_weight:.2f}")

# ==========================================
# 4. Submission Phase (최종 학습 및 예측)
# ==========================================
print("\n========== Final Submission Phase ==========")

print("Retraining SVD++ on ALL data...")
final_svd_model = run_svdpp(train_df, full_train=True)

print("Retraining CatBoost on ALL data...")
final_cat_model = run_catboost(train_df, full_train=True)

print("Predicting Test data...")
# SVD 예측
svd_test_preds = []
for _, row in test_df.iterrows():
    svd_test_preds.append(final_svd_model.predict(row['userId'], row['movieId']).est)
svd_test_preds = np.array(svd_test_preds)

# CatBoost 예측
cat_test_features = test_df[['userId', 'movieId', 'year', 'genres']]
cat_test_preds = final_cat_model.predict(cat_test_features)

# 앙상블 적용
raw_final_preds = (best_weight * svd_test_preds) + ((1 - best_weight) * cat_test_preds)

# ---------------------------------------------------------
# [중요] 0.5 단위 반올림 적용 로직
# ---------------------------------------------------------
# 1. 2를 곱함 -> 반올림 -> 2로 나눔 (예: 3.2 -> 6.4 -> 6.0 -> 3.0 / 3.3 -> 6.6 -> 7.0 -> 3.5)
final_preds = np.round(raw_final_preds * 2) / 2

# 2. 값 범위 제한 (0.5 ~ 5.0)
final_preds = np.clip(final_preds, 0.5, 5.0)
# ---------------------------------------------------------

# ==========================================
# 5. 결과 파일 저장 (rId, rating)
# ==========================================

submission_df = pd.DataFrame()
submission_df['rId'] = np.arange(1, len(test_df) + 1) # 1부터 시작하는 ID 생성
submission_df['rating'] = final_preds

submission_df.to_csv('submission_ensemble.csv', index=False)

print("\nSubmission file saved: submission_ensemble.csv")
print(submission_df.head(10)) # 상위 10개 출력하여 0.5 단위 확인