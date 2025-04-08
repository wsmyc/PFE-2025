import torch
import torch.nn as nn

class Recommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        # Validation des indices
        if (user_ids >= self.num_users).any() or (item_ids >= self.num_items).any():
            invalid_users = user_ids[user_ids >= self.num_users].unique()
            invalid_items = item_ids[item_ids >= self.num_items].unique()
            raise ValueError(f"IDs hors limites - Utilisateurs: {invalid_users}, Articles: {invalid_items}")
            
        users = self.user_emb(user_ids)
        items = self.item_emb(item_ids)
        bias = self.user_bias(user_ids) + self.item_bias(item_ids)
        return (users * items).sum(dim=1) + bias.squeeze()

    def pearson_similarity(self, user_matrix):
        normalized = user_matrix - user_matrix.mean(dim=1, keepdim=True)
        numerator = torch.mm(normalized, normalized.T)
        std = torch.sqrt(torch.sum(normalized**2, dim=1))
        denominator = torch.outer(std, std) + 1e-8
        return numerator / denominator