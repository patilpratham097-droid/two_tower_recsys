import torch
import pandas as pd
import os
from src.model import TwoTowerModel
from src.dataset import load_genres_matrix

def load_movie_titles():
    extract_path = os.path.join("data", "ml-100k")
    item_file = os.path.join(extract_path, "u.item")
    if not os.path.exists(item_file):
        item_file = os.path.join(extract_path, "ml-100k", "u.item")

    movies_df = pd.read_csv(item_file, sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])
    return movies_df

def recommend_movies(user_id_raw, top_k=5):
    print(f"\nGenerating top {top_k} hybrid recommendations for User {user_id_raw}...")
    
    extract_path = os.path.join("data", "ml-100k")
    ratings_file = os.path.join(extract_path, "u.data")
    if not os.path.exists(ratings_file):
        ratings_file = os.path.join(extract_path, "ml-100k", "u.data")

    df = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()
    
    # Load genre metadata matrix
    item_genres_np = load_genres_matrix(extract_path, num_items)
    item_genres_tensor = torch.tensor(item_genres_np, dtype=torch.float32)

    # Load trained model
    model = TwoTowerModel(num_users=num_users, num_items=num_items, embedding_dim=32, num_genres=19)
    model.load_state_dict(torch.load("models/two_tower_model.pth"))
    model.eval()

    user_id = user_id_raw - 1
    if user_id < 0 or user_id >= num_users:
        print(f"Error: User ID must be between 1 and {num_users}")
        return

    with torch.no_grad():
        all_item_ids = torch.arange(num_items)
        
        # Extract item embeddings and concatenate with genre features for ALL movies
        i_embed = model.item_embedding(all_item_ids)
        item_combined = torch.cat([i_embed, item_genres_tensor], dim=1)
        item_vectors = model.item_fc(item_combined)  # Shape: [num_items, embedding_dim]

        # Extract user vector
        user_tensor = torch.tensor([user_id], dtype=torch.long)
        u_embed = model.user_embedding(user_tensor)
        user_vector = model.user_fc(u_embed)      # Shape: [1, embedding_dim]

        # Compute dot product similarity scores
        scores = torch.sum(user_vector * item_vectors, dim=1)
        top_scores, top_item_indices = torch.topk(scores, k=top_k)

    movies_df = load_movie_titles()
    
    print("-" * 40)
    print(f"Top Hybrid Recommendations:")
    print("-" * 40)
    for i in range(top_k):
        movie_idx = top_item_indices[i].item()
        score = top_scores[i].item()
        title = movies_df.loc[movies_df['item_id'] == (movie_idx + 1), 'title'].values[0]
        print(f"{i+1}. {title} (Score: {score:.3f})")
    print("-" * 40)

if __name__ == "__main__":
    recommend_movies(user_id_raw=42, top_k=5)