import numpy as np
import os

source_file = '/data1/dnian/MiniOneRec/data/Amazon18/Office/Office.emb-qwen-td-interval.npy'
target_file = '/data1/dnian/MiniOneRec/data/Amazon18/Office/Office.item2new_item_interval.npy'

print(f"Loading {source_file}...")
try:
    data = np.load(source_file, allow_pickle=True)
except Exception as e:
    print(f"Error loading file: {e}")
    exit(1)

print(f"Source shape: {data.shape}")

# Ensure we have at least 5 columns
if data.shape[1] < 5:
    print("Error: Source file has fewer than 5 columns.")
    exit(1)

# Extract columns
# New Col 1: Source Column 3 (index 2)
col1 = data[:, 2]

# New Col 2: Source Column 5 (index 4)
col2 = data[:, 4]

# New Col 3: Row Index (0 to N-1)
col3 = np.arange(len(data))

# Stack column-wise
new_data = np.column_stack((col1, col2, col3))

print(f"New data shape: {new_data.shape}")
print("First 5 rows:")
print(new_data[:5])
print("Last 5 rows:")
print(new_data[-5:])

# Save
np.save(target_file, new_data)
print(f"Saved to {target_file}")
