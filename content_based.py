import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split # 데이터 분할용

# ==========================================
# 1. 데이터셋 정의 (Dataset)
# ==========================================
class ContentBasedDataset(Dataset):
    def __init__(self, user_ids, genre_lists, years, ratings=None, max_genres=10):
        self.user_ids = torch.LongTensor(user_ids)
        self.years = torch.FloatTensor(years)
        self.ratings = torch.FloatTensor(ratings) if ratings is not None else None
        self.max_genres = max_genres
        self.genre_tensors = np.zeros((len(user_ids), max_genres), dtype=int)
        for i, genres in enumerate(genre_lists):
            length = min(len(genres), max_genres)
            if length > 0:
                self.genre_tensors[i, :length] = genres[:length]
        self.genre_tensors = torch.LongTensor(self.genre_tensors)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if self.ratings is not None:
            return self.user_ids[idx], self.genre_tensors[idx], self.years[idx], self.ratings[idx]
        else:
            return self.user_ids[idx], self.genre_tensors[idx], self.years[idx]

# ==========================================
# 2. 추천 모델 정의 (Neural Content-based Filtering)
# ==========================================
class ContentRecommender(nn.Module):
    def __init__(self, num_users, num_genres, embedding_dim, hidden_dim, dropout):
        super(ContentRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim, padding_idx=0)
        input_dim = embedding_dim + embedding_dim + 1
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout), # Dropout 파라미터 적용
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.genre_embedding.weight)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                layer.bias.data.fill_(0.0)

    def forward(self, user_idx, genre_indices, year):
        user_vec = self.user_embedding(user_idx)
        genre_vecs = self.genre_embedding(genre_indices)
        item_genre_vec = genre_vecs.mean(dim=1)
        year_vec = year.unsqueeze(1)
        combined_vec = torch.cat([user_vec, item_genre_vec, year_vec], dim=1)
        return self.fc_layers(combined_vec).squeeze()

# ==========================================
# 3. 데이터 전처리 클래스 (Pure NumPy/Pandas)
# ==========================================
class DataProcessor:
    def __init__(self):
        self.user_to_idx = {}
        self.genre_to_idx = {'<PAD>': 0}
        self.year_min = 0
        self.year_max = 1
        
    def fit(self, df_train, df_test):
        all_users = pd.concat([df_train['userId'], df_test['userId']]).unique()
        self.user_to_idx = {uid: i for i, uid in enumerate(all_users)}
        
        all_genres = set()
        for genres_str in pd.concat([df_train['genres'], df_test['genres']]):
            if pd.isna(genres_str): continue
            all_genres.update(str(genres_str).split('|'))
        
        for i, genre in enumerate(sorted(all_genres), start=1):
            self.genre_to_idx[genre] = i
            
        all_years = pd.concat([df_train['year'], df_test['year']]).values
        self.year_min = all_years.min()
        self.year_max = all_years.max()
        
    def transform(self, df):
        user_ids = df['userId'].map(lambda x: self.user_to_idx.get(x, 0)).values
        genre_lists = []
        for genres_str in df['genres']:
            indices = []
            if not pd.isna(genres_str):
                for g in str(genres_str).split('|'):
                    if g in self.genre_to_idx:
                        indices.append(self.genre_to_idx[g])
            genre_lists.append(indices)
        years = df['year'].values.astype(float)
        if self.year_max > self.year_min:
            years = (years - self.year_min) / (self.year_max - self.year_min)
        else:
            years = np.zeros_like(years)
        ratings = df['rating'].values if 'rating' in df.columns else None
        return user_ids, genre_lists, years, ratings
    
    @property
    def num_users(self): return len(self.user_to_idx)
    @property
    def num_genres(self): return len(self.genre_to_idx)

# ==========================================
# 4. Main 실행 함수
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    
    # 튜닝할 파라미터들을 인자로 받도록 설정
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()

    # 랜덤 시드 고정 (재현성을 위해 필수)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Settings: Emb={args.emb_dim}, Hidden={args.hidden_dim}, LR={args.lr}, Drop={args.dropout}, Batch={args.batch_size}")

    # 1. 데이터 로드 및 전처리
    df_train_full = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)
    
    # Train 데이터 내에서 검증셋 분리 (8:2)
    df_train, df_val = train_test_split(df_train_full, test_size=0.2, random_state=args.seed)
    
    processor = DataProcessor()
    processor.fit(df_train_full, df_test) # fit은 전체 데이터로 해야 ID 누락 방지
    
    train_data = processor.transform(df_train)
    val_data = processor.transform(df_val)
    test_data = processor.transform(df_test)
    
    train_dataset = ContentBasedDataset(*train_data)
    val_dataset = ContentBasedDataset(*val_data)
    test_dataset = ContentBasedDataset(*test_data[:3], ratings=None) # Test는 rating 없음
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. 모델 초기화
    model = ContentRecommender(
        num_users=processor.num_users,
        num_genres=processor.num_genres,
        embedding_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 3. 학습 및 검증
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for u, g, y, r in train_loader:
            u, g, y, r = u.to(DEVICE), g.to(DEVICE), y.to(DEVICE), r.to(DEVICE)
            optimizer.zero_grad()
            preds = model(u, g, y)
            loss = criterion(preds, r)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, g, y, r in val_loader:
                u, g, y, r = u.to(DEVICE), g.to(DEVICE), y.to(DEVICE), r.to(DEVICE)
                preds = model(u, g, y)
                loss = criterion(preds, r)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Validation Loss가 가장 낮을 때의 성능 기록
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 여기서 모델 저장 가능 (torch.save)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # 4. 전체 데이터로 재학습 (Optional)
    # 실험이 끝나고 최종 제출용을 만들 때는 Validation 없이 전체 데이터로 학습하는 것이 좋음
    # 여기서는 생략하고 바로 Test 예측 진행

    # 5. Test 예측 및 저장
    model.eval()
    predictions = []
    with torch.no_grad():
        for u, g, y in test_loader:
            u, g, y = u.to(DEVICE), g.to(DEVICE), y.to(DEVICE)
            preds = model(u, g, y)
            predictions.extend(preds.cpu().numpy())
            
    # 후처리 (정수 변환)
    min_rating = df_train_full['rating'].min()
    max_rating = df_train_full['rating'].max()
    predictions = np.clip(predictions, min_rating, max_rating)
    predictions = np.round(predictions*2)/2
    
    # 저장
    if 'rId' in df_test.columns:
        sub = pd.DataFrame({'rId': df_test['rId'], 'rating': predictions})
    else:
        sub = pd.DataFrame({'userId': df_test['userId'], 'movieId': df_test['movieId'], 'rating': predictions})
        
    sub.to_csv('submission.csv', index=False)
    print("submission.csv saved.")

if __name__ == "__main__":
    main()