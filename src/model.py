import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, num_genres=19):
        super(TwoTowerModel, self).__init__()
        
        # User Tower
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # Item Tower (combines ID embedding + 19 genre features)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        self.item_fc = nn.Sequential(
            nn.Linear(embedding_dim + num_genres, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, user_ids, item_ids, item_genres):
        # User vector path
        u_embed = self.user_embedding(user_ids)
        user_vector = self.user_fc(u_embed)

        # Item vector path (concatenate ID embedding with explicit genre flags)
        i_embed = self.item_embedding(item_ids)
        item_combined = torch.cat([i_embed, item_genres], dim=1)
        item_vector = self.item_fc(item_combined)

        # Dot product similarity
        interactions = torch.sum(user_vector * item_vector, dim=1)
        return torch.sigmoid(interactions)