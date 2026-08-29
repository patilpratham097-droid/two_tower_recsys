import os
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df, item_genres):
        self.user_ids = torch.tensor(ratings_df['user_id'].values, dtype=torch.long)
        self.item_ids = torch.tensor(ratings_df['item_id'].values, dtype=torch.long)
        self.labels = torch.tensor((ratings_df['rating'] >= 4.0).values, dtype=torch.float32)
        self.item_genres = torch.tensor(item_genres, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        u_id = self.user_ids[idx]
        i_id = self.item_ids[idx]
        label = self.labels[idx]
        genres = self.item_genres[i_id] # Fetch genre features for this specific movie
        return u_id, i_id, genres, label

def load_genres_matrix(extract_path, num_items):
    item_file = os.path.join(extract_path, "u.item")
    if not os.path.exists(item_file):
        item_file = os.path.join(extract_path, "ml-100k", "u.item")

    # u.item has 24 columns: 0-1 are ID and Title, 2-4 are URLs/dates, 5-23 are the 19 genre flags
    cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + [f'genre_{i}' for i in range(19)]
    movies_meta = pd.read_csv(item_file, sep='|', encoding='latin-1', header=None, names=cols)
    
    # Extract only the 19 genre columns as a numpy matrix
    genre_matrix = movies_meta[[f'genre_{i}' for i in range(19)]].values
    return genre_matrix

def get_data_loaders(batch_size=256):
    data_dir = "data"
    extract_path = os.path.join(data_dir, "ml-100k")
    ratings_file = os.path.join(extract_path, "u.data")
    if not os.path.exists(ratings_file):
        ratings_file = os.path.join(extract_path, "ml-100k", "u.data")

    df = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    df['user_id'] = df['user_id'] - 1
    df['item_id'] = df['item_id'] - 1

    item_genres = load_genres_matrix(extract_path, df['item_id'].nunique())

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_dataset = MovieLensDataset(train_df, item_genres)
    test_dataset = MovieLensDataset(test_df, item_genres)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_users = df['user_id'].nunique()
    num_items = df['item_id'].nunique()

    return train_loader, test_loader, num_users, num_items, item_genres