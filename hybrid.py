import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Dataset (ID + Content 모두 포함)
# ==========================================
class HybridDataset(Dataset):
    def __init__(self, user_ids, item_ids, genre_lists, years, ratings=None, max_genres=10):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.years = torch.FloatTensor(years)
        self.ratings = torch.FloatTensor(ratings) if ratings is not None else None
        
        # Genre Padding
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
            return self.user_ids[idx], self.item_ids[idx], self.genre_tensors[idx], self.years[idx], self.ratings[idx]
        else:
            return self.user_ids[idx], self.item_ids[idx], self.genre_tensors[idx], self.years[idx]

# ==========================================
# 2. Hybrid Model (CF + Content)
# ==========================================
class HybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, num_genres, emb_dim=32, hidden_dim=64, dropout=0.2):
        super(HybridRecommender, self).__init__()
        
        # --- CF Parts (ID Embeddings) ---
        self.user_id_embedding = nn.Embedding(num_users, emb_dim)
        self.item_id_embedding = nn.Embedding(num_items, emb_dim)
        
        # --- Content Parts (Feature Embeddings) ---
        self.genre_embedding = nn.Embedding(num_genres, emb_dim, padding_idx=0)
        
        # MLP Input Size 계산:
        # UserID(32) + ItemID(32) + GenreMean(32) + Year(1)
        input_dim = emb_dim + emb_dim + emb_dim + 1
        
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 학습 안정화
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        nn.init.xavier_uniform_(self.genre_embedding.weight)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, user_idx, item_idx, genre_indices, year):
        # 1. ID Embedding (CF signal)
        u_vec = self.user_id_embedding(user_idx)
        i_vec = self.item_id_embedding(item_idx)
        
        # 2. Content Embedding (CB signal)
        g_vecs = self.genre_embedding(genre_indices)
        g_vec = g_vecs.mean(dim=1) # 장르 평균
        
        y_vec = year.unsqueeze(1)
        
        # 3. Concatenate Everything
        # [유저ID특성, 아이템ID특성, 장르특성, 연도]를 모두 합침
        vector = torch.cat([u_vec, i_vec, g_vec, y_vec], dim=1)
        
        # 4. Predict
        pred = self.fc_layers(vector)
        return pred.squeeze()

# ==========================================
# 3. Data Processor
# ==========================================
class DataProcessor:
    def __init__(self):
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.genre_to_idx = {'<PAD>': 0}
        self.year_min = 0
        self.year_max = 1
        
    def fit(self, df_train, df_test):
        # ID Mappings
        all_users = pd.concat([df_train['userId'], df_test['userId']]).unique()
        self.user_to_idx = {uid: i for i, uid in enumerate(all_users)}
        
        all_items = pd.concat([df_train['movieId'], df_test['movieId']]).unique()
        self.item_to_idx = {iid: i for i, iid in enumerate(all_items)}
        
        # Genre Mapping
        all_genres = set()
        for genres_str in pd.concat([df_train['genres'], df_test['genres']]):
            if pd.isna(genres_str): continue
            all_genres.update(str(genres_str).split('|'))
        for i, genre in enumerate(sorted(all_genres), start=1):
            self.genre_to_idx[genre] = i
            
        # Year Statistics
        all_years = pd.concat([df_train['year'], df_test['year']]).values
        self.year_min = all_years.min()
        self.year_max = all_years.max()

    def transform(self, df):
        # IDs
        user_ids = df['userId'].map(lambda x: self.user_to_idx.get(x, 0)).values
        item_ids = df['movieId'].map(lambda x: self.item_to_idx.get(x, 0)).values # Unknown은 0번 처리
        
        # Genres
        genre_lists = []
        for genres_str in df['genres']:
            indices = []
            if not pd.isna(genres_str):
                for g in str(genres_str).split('|'):
                    if g in self.genre_to_idx:
                        indices.append(self.genre_to_idx[g])
            genre_lists.append(indices)
            
        # Year
        years = df['year'].values.astype(float)
        if self.year_max > self.year_min:
            years = (years - self.year_min) / (self.year_max - self.year_min)
        else:
            years = np.zeros_like(years)
            
        ratings = df['rating'].values if 'rating' in df.columns else None
        
        return user_ids, item_ids, genre_lists, years, ratings
    
    @property
    def num_users(self): return len(self.user_to_idx)
    @property
    def num_items(self): return len(self.item_to_idx)
    @property
    def num_genres(self): return len(self.genre_to_idx)

# ==========================================
# 4. Main 실행
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    
    # 파라미터 튜닝 영역
    parser.add_argument('--emb_dim', type=int, default=32) 
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20) # Epoch 수 증가 추천
    
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hybrid Recommender on {DEVICE}")

    # 1. Data Load
    df_train_full = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)
    
    df_train, df_val = train_test_split(df_train_full, test_size=0.1, random_state=42)
    
    processor = DataProcessor()
    processor.fit(df_train_full, df_test)
    
    train_data = processor.transform(df_train)
    val_data = processor.transform(df_val)
    test_data = processor.transform(df_test)
    
    train_dataset = HybridDataset(*train_data)
    val_dataset = HybridDataset(*val_data)
    test_dataset = HybridDataset(*test_data[:4], ratings=None)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. Model
    model = HybridRecommender(
        num_users=processor.num_users,
        num_items=processor.num_items,
        num_genres=processor.num_genres,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 3. Training
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for u, i, g, y, r in train_loader:
            u, i, g, y, r = u.to(DEVICE), i.to(DEVICE), g.to(DEVICE), y.to(DEVICE), r.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(u, i, g, y)
            loss = criterion(pred, r)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, g, y, r in val_loader:
                u, i, g, y, r = u.to(DEVICE), i.to(DEVICE), g.to(DEVICE), y.to(DEVICE), r.to(DEVICE)
                pred = model(u, i, g, y)
                val_loss += criterion(pred, r).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val

    # 4. Prediction
    model.eval()
    preds = []
    
    min_rating = df_train_full['rating'].min()
    max_rating = df_train_full['rating'].max()
    
    with torch.no_grad():
        for u, i, g, y in test_loader:
            u, i, g, y = u.to(DEVICE), i.to(DEVICE), g.to(DEVICE), y.to(DEVICE)
            pred = model(u, i, g, y)
            preds.extend(pred.cpu().numpy())
            
    # Post-processing
    preds = np.array(preds)
    preds = np.clip(preds, min_rating, max_rating)
    preds = np.round(preds * 2) / 2
    
    if 'rId' in df_test.columns:
        submission = pd.DataFrame({'rId': df_test['rId'], 'rating': preds})
    else:
        submission = pd.DataFrame({'userId': df_test['userId'], 'movieId': df_test['movieId'], 'rating': preds})
        
    submission.to_csv('submission.csv', index=False)
    print("Hybrid Submission Saved!")

if __name__ == "__main__":
    main()