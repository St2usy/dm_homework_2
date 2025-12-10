import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Dataset 정의
# ==========================================
class CFDataset(Dataset):
    def __init__(self, users, items, ratings=None):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings) if ratings is not None else None

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        if self.ratings is not None:
            return self.users[idx], self.items[idx], self.ratings[idx]
        else:
            return self.users[idx], self.items[idx]

# ==========================================
# 2. Matrix Factorization 모델 정의
# ==========================================
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, use_bias=False):
        super(MatrixFactorization, self).__init__()
        self.use_bias = use_bias # 바이어스 사용 여부 플래그
        
        # 1. Embeddings (User & Item Vector) - 이건 무조건 필요
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 2. Bias Terms (옵션에 따라 생성)
        if self.use_bias:
            self.user_bias = nn.Embedding(num_users, 1)
            self.item_bias = nn.Embedding(num_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        if self.use_bias:
            self.user_bias.weight.data.fill_(0.0)
            self.item_bias.weight.data.fill_(0.0)

    def forward(self, user, item):
        # (Batch, Dim)
        u_emb = self.user_embedding(user)
        i_emb = self.item_embedding(item)
        
        # Dot Product (순수 상호작용)
        interaction = (u_emb * i_emb).sum(dim=1)
        
        # Global Effect가 없을 때 (Bias 제거)
        if not self.use_bias:
            return interaction
            
        # Global Effect가 있을 때 (기존 방식)
        else:
            u_b = self.user_bias(user).squeeze()
            i_b = self.item_bias(item).squeeze()
            return self.global_bias + u_b + i_b + interaction

# ==========================================
# 3. 데이터 처리 클래스
# ==========================================
class DataProcessor:
    def __init__(self):
        self.user_to_idx = {}
        self.item_to_idx = {}
        
    def fit(self, df_train, df_test):
        # Train과 Test에 있는 모든 ID를 수집하여 매핑 테이블 생성
        # (Collaborative Filtering은 ID 기반이므로 ID 매핑이 가장 중요함)
        all_users = pd.concat([df_train['userId'], df_test['userId']]).unique()
        all_items = pd.concat([df_train['movieId'], df_test['movieId']]).unique()
        
        self.user_to_idx = {uid: i for i, uid in enumerate(all_users)}
        self.item_to_idx = {iid: i for i, iid in enumerate(all_items)}

    def transform(self, df):
        # 매핑 테이블에 없는 ID는 0번으로 처리하거나 예외처리 해야 함
        # 여기서는 fit에서 합집합을 썼으므로 모든 ID가 존재함
        users = df['userId'].map(self.user_to_idx).values
        items = df['movieId'].map(self.item_to_idx).values
        ratings = df['rating'].values if 'rating' in df.columns else None
        
        return users, items, ratings
    
    @property
    def num_users(self): return len(self.user_to_idx)
    
    @property
    def num_items(self): return len(self.item_to_idx)

# ==========================================
# 4. Main 실행 함수
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    
    # 튜닝 가능한 파라미터
    parser.add_argument('--emb_dim', type=int, default=20) # 보통 20~100 사이
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--no_bias', action='store_true', help='Use this flag to disable bias terms')
    
    args = parser.parse_args()

    # 시드 고정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    print(f"Settings: Emb={args.emb_dim}, LR={args.lr}, Batch={args.batch_size}, Global_effact={not args.no_bias}")

    # 1. 데이터 로드
    df_train_full = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)
    
    # Validation Split
    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=args.seed)
    
    # 2. ID 매핑
    processor = DataProcessor()
    processor.fit(df_train_full, df_test) # 전체 데이터 기준 매핑
    
    train_users, train_items, train_ratings = processor.transform(df_train)
    val_users, val_items, val_ratings = processor.transform(df_val)
    test_users, test_items, _ = processor.transform(df_test)
    
    # 3. Dataset & DataLoader
    train_dataset = CFDataset(train_users, train_items, train_ratings)
    val_dataset = CFDataset(val_users, val_items, val_ratings)
    test_dataset = CFDataset(test_users, test_items)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    use_bias_flag = not args.no_bias
    
    # 4. 모델 초기화
    model = MatrixFactorization(
        num_users=processor.num_users, 
        num_items=processor.num_items, 
        embedding_dim=args.emb_dim,
        use_bias=use_bias_flag
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    # Weight Decay는 과적합 방지를 위해 중요 (L2 Regularization 역할)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 5. 학습 Loop
    print(f"Start Training CF Model (Factors={args.emb_dim})...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for u, i, r in train_loader:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            
            optimizer.zero_grad()
            prediction = model(u, i)
            loss = criterion(prediction, r)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, r in val_loader:
                u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
                prediction = model(u, i)
                loss = criterion(prediction, r)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            # 모델 저장 로직 추가 가능
    print(f"Best Validation Loss: {best_loss:.4f}")

    # 6. 예측 및 0.5 단위 후처리
    print("Predicting...")
    model.eval()
    predictions = []
    
    # Clipping을 위한 범위 설정
    min_rating = df_train_full['rating'].min()
    max_rating = df_train_full['rating'].max()
    
    with torch.no_grad():
        for u, i in test_loader:
            u, i = u.to(DEVICE), i.to(DEVICE)
            pred = model(u, i)
            predictions.extend(pred.cpu().numpy())
            
    predictions = np.array(predictions)
    
    # 값 범위 제한 (1.0 ~ 5.0)
    predictions = np.clip(predictions, min_rating, max_rating)
    # 0.5 단위 반올림
    predictions = np.round(predictions * 2) / 2
    
    # 저장
    if 'rId' in df_test.columns:
        submission = pd.DataFrame({'rId': df_test['rId'], 'rating': predictions})
    else:
        submission = pd.DataFrame({'userId': df_test['userId'], 'movieId': df_test['movieId'], 'rating': predictions})
        
    submission.to_csv('submission1.csv', index=False)
    print("CF Submission Saved!")

if __name__ == "__main__":
    main()