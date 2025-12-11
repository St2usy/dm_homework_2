import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import copy

# ==========================================
# 1. Dataset (Field-aware 구조로 변경)
# ==========================================
class DeepFMDataset(Dataset):
    def __init__(self, user_ids, item_ids, genre_lists, years, ratings=None, max_genres=3):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.years = torch.FloatTensor(years)
        self.ratings = torch.FloatTensor(ratings) if ratings is not None else None
        
        # 장르 처리 (Multi-hot을 위한 단순화: 가장 앞의 max_genres개만 사용하거나 임의 선택)
        # DeepFM은 입력 필드 개수가 고정되는 것이 유리하므로, 주요 장르 1~3개만 피처로 사용
        self.genre_indices = np.zeros((len(user_ids), max_genres), dtype=int)
        for i, genres in enumerate(genre_lists):
            length = min(len(genres), max_genres)
            if length > 0:
                self.genre_indices[i, :length] = genres[:length]
        self.genre_indices = torch.LongTensor(self.genre_indices)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # 리턴: (User, Item, Genre1, Genre2, Genre3, Year), Rating
        if self.ratings is not None:
            return self.user_ids[idx], self.item_ids[idx], self.genre_indices[idx], self.years[idx], self.ratings[idx]
        else:
            return self.user_ids[idx], self.item_ids[idx], self.genre_indices[idx], self.years[idx]

# ==========================================
# 2. DeepFM Model Definition
# ==========================================
class DeepFM(nn.Module):
    def __init__(self, num_users, num_items, num_genres, emb_dim=16, hidden_dims=[64, 32], dropout=0.2):
        super(DeepFM, self).__init__()
        
        # ---------------------------------------------------------
        # A. Feature Sizes
        # ---------------------------------------------------------
        # 입력으로 들어오는 카테고리형 변수의 개수 (User + Item + Genre*3) = 5개 필드
        self.num_categorical_fields = 2 + 3 
        
        # ---------------------------------------------------------
        # B. Embeddings (Shared between FM and Deep parts)
        # ---------------------------------------------------------
        # 모든 피처는 같은 차원(emb_dim)을 가져야 FM 연산이 가능함
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.genre_emb = nn.Embedding(num_genres, emb_dim, padding_idx=0)
        
        # Bias terms for Linear part (1차원)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.genre_bias = nn.Embedding(num_genres, 1, padding_idx=0)
        
        # Year는 연속형 변수이므로 Linear Layer로 임베딩 차원만큼 확장
        self.year_emb = nn.Linear(1, emb_dim)
        
        self._init_weights()

        # ---------------------------------------------------------
        # C. Deep Component (MLP)
        # ---------------------------------------------------------
        # 입력 차원: (카테고리 필드 수 + Year 1개) * Emb_dim
        input_dim = (self.num_categorical_fields + 1) * emb_dim
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        
        self.deep_layers = nn.Sequential(*layers)

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        nn.init.xavier_normal_(self.genre_emb.weight)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.item_bias.weight, 0)
        nn.init.constant_(self.genre_bias.weight, 0)

    def forward(self, user, item, genres, year):
        # ---------------------------------------------------------
        # 1. Embedding Lookup
        # ---------------------------------------------------------
        # (Batch, Emb_Dim)
        u_e = self.user_emb(user)
        i_e = self.item_emb(item)
        # Genre는 (Batch, 3, Emb_Dim) -> Flatten해서 쓰는게 아니라 필드별로 쪼개거나 Sum해서 처리
        # 여기서는 DeepFM의 정석대로 필드별로 취급하기 위해 각각 나눔
        g1_e = self.genre_emb(genres[:, 0])
        g2_e = self.genre_emb(genres[:, 1])
        g3_e = self.genre_emb(genres[:, 2])
        
        # Year (Batch, 1) -> (Batch, Emb_Dim)
        y_e = self.year_emb(year.unsqueeze(1))
        
        # 모든 임베딩을 스택 (Batch, Num_Fields, Emb_Dim)
        # Fields: User, Item, G1, G2, G3, Year
        stacked_emb = torch.stack([u_e, i_e, g1_e, g2_e, g3_e, y_e], dim=1)
        
        # ---------------------------------------------------------
        # 2. FM Component (Factorization Machine)
        # ---------------------------------------------------------
        # 공식: 0.5 * [ (Sum of embeddings)^2 - Sum of (embeddings^2) ]
        # 이 수식이 모든 피처 쌍(Pair)의 내적 합을 효율적으로 계산함
        
        sum_of_emb = torch.sum(stacked_emb, dim=1) # (Batch, Emb_Dim)
        sum_of_sq_emb = torch.sum(stacked_emb**2, dim=1) # (Batch, Emb_Dim)
        
        # (Batch, Emb_Dim) -> Sum -> (Batch, 1)
        fm_out = 0.5 * torch.sum(sum_of_emb**2 - sum_of_sq_emb, dim=1, keepdim=True)
        
        # ---------------------------------------------------------
        # 3. Linear Component (1st Order)
        # ---------------------------------------------------------
        # Bias들의 합 + Year 자체 값
        u_b = self.user_bias(user)
        i_b = self.item_bias(item)
        g_b = self.genre_bias(genres).sum(dim=1)
        # Year는 bias가 따로 없으므로 w*x 형태로 가정 (여기선 생략하거나 Linear 결과 사용)
        
        linear_out = u_b + i_b + g_b # (Batch, 1)
        
        # ---------------------------------------------------------
        # 4. Deep Component (DNN)
        # ---------------------------------------------------------
        # Flatten (Batch, Fields * Emb_Dim)
        dnn_input = stacked_emb.view(stacked_emb.size(0), -1)
        deep_out = self.deep_layers(dnn_input)
        
        # ---------------------------------------------------------
        # 5. Final Sum
        # ---------------------------------------------------------
        return (linear_out + fm_out + deep_out).squeeze()

# ==========================================
# 3. Data Processor (기존과 유사)
# ==========================================
class DataProcessor:
    def __init__(self):
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.genre_to_idx = {'<PAD>': 0}
        self.year_min = 0; self.year_max = 1
        
    def fit(self, df_train, df_test):
        all_users = pd.concat([df_train['userId'], df_test['userId']]).unique()
        self.user_to_idx = {uid: i for i, uid in enumerate(all_users)}
        
        all_items = pd.concat([df_train['movieId'], df_test['movieId']]).unique()
        self.item_to_idx = {iid: i for i, iid in enumerate(all_items)}
        
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
        item_ids = df['movieId'].map(lambda x: self.item_to_idx.get(x, 0)).values
        
        genre_lists = []
        for genres_str in df['genres']:
            indices = []
            if not pd.isna(genres_str):
                for g in str(genres_str).split('|'):
                    if g in self.genre_to_idx: indices.append(self.genre_to_idx[g])
            genre_lists.append(indices)
            
        years = df['year'].values.astype(float)
        years = (years - self.year_min) / (self.year_max - self.year_min) if self.year_max > self.year_min else np.zeros_like(years)
        ratings = df['rating'].values if 'rating' in df.columns else None
        
        return user_ids, item_ids, genre_lists, years, ratings
    
    @property
    def num_users(self): return len(self.user_to_idx)
    @property
    def num_items(self): return len(self.item_to_idx)
    @property
    def num_genres(self): return len(self.genre_to_idx)

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--test', type=str, required=True)
    
    # DeepFM Recommended Hyperparameters
    parser.add_argument('--emb_dim', type=int, default=16) # DeepFM은 임베딩이 너무 크면 FM 파트에서 노이즈가 심해짐
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-4) # 규제 필수
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DeepFM on {DEVICE}")

    print(f"Settings: Emb={args.emb_dim}, Hidden={args.hidden_dim}, LR={args.lr}, Drop={args.dropout}, Batch={args.batch_size}")
   

    # Load Data
    df_train_full = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)
    df_train, df_val = train_test_split(df_train_full, test_size=0.1, random_state=args.seed)
    
    processor = DataProcessor()
    processor.fit(df_train_full, df_test)
    
    # Create Datasets
    train_data = processor.transform(df_train)
    val_data = processor.transform(df_val)
    test_data = processor.transform(df_test)
    
    train_dataset = DeepFMDataset(*train_data, max_genres=3)
    val_dataset = DeepFMDataset(*val_data, max_genres=3)
    test_dataset = DeepFMDataset(*test_data[:4], max_genres=3, ratings=None)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize Model
    model = DeepFM(
        num_users=processor.num_users,
        num_items=processor.num_items,
        num_genres=processor.num_genres,
        emb_dim=args.emb_dim,
        hidden_dims=[args.hidden_dim, args.hidden_dim//2],
        dropout=args.dropout
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Training Loop
    best_loss = float('inf')
    best_epoch = 0
    # best_model_state = None
    
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
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, g, y, r in val_loader:
                u, i, g, y, r = u.to(DEVICE), i.to(DEVICE), g.to(DEVICE), y.to(DEVICE), r.to(DEVICE)
                pred = model(u, i, g, y)
                val_loss += criterion(pred, r).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {current_lr}")
        
        scheduler.step(avg_val)
        
        # Save Best Model
        if avg_val < best_loss:
            best_loss = avg_val
            # best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

    # Prediction using Best Model
    print(f"Best Val Loss: {best_loss:.4f} at Epoch {best_epoch}")
    
    print(f"\n>>> Phase 2: Re-training on FULL Dataset for {best_epoch} epochs...")
    
    # 1. 전체 데이터로 데이터셋/로더 다시 생성
    full_train_data = processor.transform(df_train_full)
    full_train_dataset = DeepFMDataset(*full_train_data, max_genres=3)
    full_train_loader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. 모델 및 옵티마이저 초기화 (완전히 새롭게 시작)
    final_model = DeepFM(
        num_users=processor.num_users,
        num_items=processor.num_items,
        num_genres=processor.num_genres,
        emb_dim=args.emb_dim,
        hidden_dims=[args.hidden_dim, args.hidden_dim//2],
        dropout=args.dropout
    ).to(DEVICE)
    
    final_optimizer = optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Full Train에서는 Validation이 없으므로 Scheduler는 보통 끄거나 동일하게 적용. 여기선 단순화를 위해 끔.
    
    # 3. Best Epoch만큼 재학습
    for epoch in range(best_epoch):
        final_model.train()
        total_loss = 0
        for u, i, g, y, r in full_train_loader:
            u, i, g, y, r = u.to(DEVICE), i.to(DEVICE), g.to(DEVICE), y.to(DEVICE), r.to(DEVICE)
            final_optimizer.zero_grad()
            pred = final_model(u, i, g, y)
            loss = criterion(pred, r)
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(full_train_loader)
        print(f"Full Train Epoch {epoch+1}/{best_epoch} | Loss: {avg_loss:.4f}")


    final_model.eval()
    preds = []
    
    min_rating = df_train_full['rating'].min()
    max_rating = df_train_full['rating'].max()
    
    with torch.no_grad():
        for u, i, g, y in test_loader:
            u, i, g, y = u.to(DEVICE), i.to(DEVICE), g.to(DEVICE), y.to(DEVICE)
            pred = final_model(u, i, g, y)
            preds.extend(pred.cpu().numpy())
            
    preds = np.array(preds)
    preds = np.clip(preds, min_rating, max_rating)
    preds = np.round(preds * 2) / 2
    
    if 'rId' in df_test.columns:
        sub = pd.DataFrame({'rId': df_test['rId'], 'rating': preds})
    else:
        sub = pd.DataFrame({'userId': df_test['userId'], 'movieId': df_test['movieId'], 'rating': preds})
    
    sub.to_csv('submission4.csv', index=False)
    print("DeepFm Submission Saved!.")

if __name__ == "__main__":
    main()