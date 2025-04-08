import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class RatingDataset(Dataset):
    def __init__(self, ratings_df, menu_df):
        self.user_ids = torch.LongTensor(ratings_df['id_utilisateur'] - 1)
        self.item_ids = torch.LongTensor([menu_df[menu_df['id_article'] == x].index[0] for x in ratings_df['id_article']])
        self.ratings = torch.FloatTensor(ratings_df['note'])

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

class Recommender(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
    def forward(self, users, items):
        users_emb = self.user_emb(users)
        items_emb = self.item_emb(items)
        bias = self.user_bias(users) + self.item_bias(items)
        return (users_emb * items_emb).sum(1) + bias.squeeze()

# Entraînement
if __name__ == "__main__":
    # Charger données
    ratings = pd.read_csv('data/notes.csv')
    menu = pd.read_csv('data/menu.csv')
    
    # Initialiser modèle
    model = Recommender(
        num_users=ratings['id_utilisateur'].nunique(),
        num_items=len(menu)
    )
    
    # Entraîner
    dataset = RatingDataset(ratings, menu)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(50):
        total_loss = 0
        for users, items, ratings in loader:
            optimizer.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(loader):.4f}")
    
    # Sauvegarder
    torch.save(model.state_dict(), 'models/bistro_model.pth')
    print("Modèle entraîné ✓")