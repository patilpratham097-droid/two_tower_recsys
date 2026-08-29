from src.dataset import get_data_loaders

print("Testing data pipeline...")
train_loader, test_loader, num_users, num_items = get_data_loaders(batch_size=256)

print(f"Successfully loaded data!")
print(f"Total Unique Users: {num_users}")
print(f"Total Unique Items (Movies): {num_items}")
print(f"Number of training batches: {len(train_loader)}")

# Grab a single batch to check what it looks like
users, items, labels = next(iter(train_loader))
print(f"Sample User IDs shape: {users.shape}")
print(f"Sample Item IDs shape: {items.shape}")
print(f"Sample Labels shape: {labels.shape}")
