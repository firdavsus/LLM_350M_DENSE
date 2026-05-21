import os
import numpy as np


train_file =  'train.bin'
test_file ='test.bin'

# 1. Calculate token counts
file_size_bytes = os.path.getsize(train_file)
total_tokens = file_size_bytes // 2  # np.uint16 is 2 bytes per token

# 0.001% is 0.00001 as a decimal
test_ratio = 0.00001
test_tokens = int(total_tokens * test_ratio)
train_tokens = total_tokens - test_tokens

test_bytes = test_tokens * 2
train_bytes = train_tokens * 2

print(f"📊 Original file size: {file_size_bytes / (1024**3):.2f} GB")
print(f"🔢 Total tokens: {total_tokens:,}")
print(f"🚆 Train tokens: {train_tokens:,}")
print(f"🧪 Test tokens:  {test_tokens:,}")

# 2. Extract and Truncate (The fast, low-disk-space way)
print(f"\n✂️ Extracting the last {test_tokens:,} tokens for test.bin...")

with open(train_file, 'r+b') as f:
    # Seek exactly to the point where the test data begins
    f.seek(train_bytes)
    
    # Read the end of the file into memory (this will be tiny, < 1 MB)
    test_data = f.read(test_bytes)
    
    # Chop off the end of the original train.bin file!
    f.truncate(train_bytes)

# 3. Save the test data
with open(test_file, 'wb') as f:
    f.write(test_data)

print(f"✅ Split complete!")
print(f"📂 Updated 'train.bin' (size: {os.path.getsize(train_file) / (1024**3):.2f} GB)")
print(f"📂 Created 'test.bin' (size: {os.path.getsize(test_file) / 1024:.2f} KB)")