import torch
import pandas as pd

class RecommenderSystem:
    def __init__(self):
        self.menu = pd.read_csv('data/menu.csv')
        self.model = torch.load('models/bistro_model.pth')
        self.model.eval()
        
    def recommend(self, user_id, top_n=5):
        # Convertir IDs
        user_tensor = torch.LongTensor([user_id - 1])
        items_tensor = torch.arange(len(self.menu)).long()
        
        # Prédictions
        with torch.no_grad():
            scores = self.model(user_tensor.repeat(len(self.menu)), items_tensor)
        
        # Top articles
        _, indices = torch.topk(scores, top_n)
        return self.menu.iloc[indices.numpy()]

# Utilisation
if __name__ == "__main__":
    rs = RecommenderSystem()
    print(rs.recommend(user_id=1))