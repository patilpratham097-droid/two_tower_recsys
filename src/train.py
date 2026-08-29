import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_data_loaders
from src.model import TwoTowerModel

def train():
    print("Initializing Enhanced Two-Tower Recommendation Training (with Genres)...")
    
    # 1. Load data loaders, dimensions, and the global genre matrix
    batch_size = 256
    train_loader, test_loader, num_users, num_items, item_genres = get_data_loaders(batch_size=batch_size)
    
    print(f"Loaded {num_users} users, {num_items} items, and genre metadata.")

    # 2. Instantiate Model, Loss, and Optimizer
    embedding_dim = 32
    # Convert global item_genres numpy array to a PyTorch tensor for inference/lookup storage
    global_item_genres_tensor = torch.tensor(item_genres, dtype=torch.float32)

    model = TwoTowerModel(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim, num_genres=19)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. Training Loop
    epochs = 5
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        for user_ids, item_ids, genres, labels in train_loader:
            optimizer.zero_grad()
            
            # Forward pass with genre features included
            predictions = model(user_ids, item_ids, genres)
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Training Loss: {avg_loss:.4f}")

    print("Training Complete!")
    
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/two_tower_model.pth")
    print("Model saved successfully to models/two_tower_model.pth")

if __name__ == "__main__":
    train()