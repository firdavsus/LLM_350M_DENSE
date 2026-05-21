import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from huggingface_hub import login
import time

login("") # <- paste your huggingface token as this dataset is massive and will probably require 10+ hours 

model_id = "EleutherAI/gpt-neox-20b"
enc = AutoTokenizer.from_pretrained(model_id, use_fast=True)
eos_id = enc.eos_token_id 

if __name__ == '__main__':
    TOTAL_ROWS_TARGET = 100_000_000 
    
    datasets_config = [
        {
            "path": "HuggingFaceFW/fineweb-edu",
            "name": "sample-10BT", 
            "text_col": "text",
            "weight": 0.60
        },
        {
            "path": "bigcode/starcoderdata",
            "data_dir": "python", 
            "text_col": "content",
            "weight": 0.20
        },
        {
            "path": "HuggingFaceTB/cosmopedia",
            "name": "stanford", 
            "text_col": "text",
            "weight": 0.10
        },
        {
            "path": "wikipedia",
            "name": "20220301.en",
            "text_col": "text",
            "weight": 0.10
        }
    ]

    # =====================================================================
    # 1. CALCULATE RESUME STATE
    # =====================================================================
    data_dir = os.path.dirname(__file__)
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, 'train.bin')
    
    tokens_already_written = 0
    estimated_rows_processed = 0
    
    if os.path.exists(filename):
        # np.uint16 is 2 bytes per token
        file_size_bytes = os.path.getsize(filename)
        tokens_already_written = file_size_bytes // 2
        
        # Estimate rows processed. If you assume ~500 tokens per row:
        estimated_rows_processed = tokens_already_written // 500
        print(f"📊 Found existing train.bin with {file_size_bytes / (1024**3):.2f} GB")
        print(f"📊 Tokens already saved: {tokens_already_written:,}")
        print(f"📊 Estimated rows already processed: {estimated_rows_processed:,}")
        
        if estimated_rows_processed >= TOTAL_ROWS_TARGET:
            print("✅ Target already reached. Exiting.")
            exit()
    else:
        print("🆕 No existing train.bin found. Starting from scratch.")

    # =====================================================================
    # 2. SETUP STREAMS
    # =====================================================================
    print(f"\n🚀 Streaming the Golden Mix...")
    streamed_iterables = []
    
    for config in datasets_config:
        print(f"🔗 Connecting to {config['path']}...")
        
        # Adding a sleep to prevent aggressive API rate limiting on setup
        time.sleep(1) 
        
        ds = load_dataset(
            path=config["path"],
            name=config.get("name"),
            data_dir=config.get("data_dir"),
            split="train",
            streaming=True,
            token=True
        )

        if config["text_col"] != "text":
            ds = ds.rename_column(config["text_col"], "text")
        
        ds = ds.select_columns(["text"])
        streamed_iterables.append(ds)

    print("\n🔀 Interleaving streams (Mixing domains)...")
    mixed_stream = interleave_datasets(
        streamed_iterables, 
        probabilities=[c["weight"] for c in datasets_config],
        seed=42, # MUST be the exact same seed as before to ensure same order
        stopping_strategy="all_exhausted"
    )
    
    mixed_stream = mixed_stream.shuffle(buffer_size=10_000, seed=42)

    # Skip the rows we've already processed!
    if estimated_rows_processed > 0:
        print(f"⏭️ Skipping the first {estimated_rows_processed:,} rows. This may take a minute...")
        mixed_stream = mixed_stream.skip(estimated_rows_processed)

    # =====================================================================
    # 3. STREAMING TOKENIZATION & WRITING (RESUMING)
    # =====================================================================
    
    print(f"✍️ Resuming appending tokens to {filename}...")
    
    # Open file in binary APPEND mode ('ab' instead of 'wb')
    with open(filename, 'ab') as f:
        token_count = tokens_already_written
        batch_text = []
        
        # Update the target for the tqdm loop to reflect remaining rows
        remaining_rows = TOTAL_ROWS_TARGET - estimated_rows_processed
        
        for i, example in tqdm(enumerate(mixed_stream), total=remaining_rows, desc="Processing Remaining"):
            if i >= remaining_rows:
                break
                
            batch_text.append(example['text'])
            
            if len(batch_text) >= 1000:
                # Add retry logic for Hugging Face timeouts
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        tokenized = enc(batch_text, add_special_tokens=False)
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e
                        print(f"\n⚠️ Tokenizer error (likely HF connection issue). Retrying in 5s... ({attempt+1}/{max_retries})")
                        time.sleep(5)
                
                all_ids = []
                for ids in tokenized['input_ids']:
                    ids.append(eos_id)
                    all_ids.extend(ids)
                
                chunk = np.array(all_ids, dtype=np.uint16)
                f.write(chunk.tobytes())
                
                token_count += len(all_ids)
                batch_text = []

        # Flush any remaining rows in the last batch
        if len(batch_text) > 0:
             tokenized = enc(batch_text, add_special_tokens=False)
             all_ids = []
             for ids in tokenized['input_ids']:
                 ids.append(eos_id)
                 all_ids.extend(ids)
             chunk = np.array(all_ids, dtype=np.uint16)
             f.write(chunk.tobytes())
             token_count += len(all_ids)

    print(f"\n✅ Finished! Total tokens in file: {token_count:,}")