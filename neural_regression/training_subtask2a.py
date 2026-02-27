import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

# ========================================
# DATASET CLASSES
# ========================================

class StateChangeDataset(Dataset):
    """Training dataset with sliding windows"""

    def __init__(self, csv_path: str, tokenizer, window_size: int = 5,
                 max_text_length: int = 128, predict_target: str = 'valence', use_words: bool = False, train_dataset: bool = True):
        """
        Args:
            predict_target: 'valence', 'arousal', or 'both'
        """
        self.predict_target = predict_target
        assert predict_target in ['valence', 'arousal', 'both'], "predict_target must be 'valence', 'arousal', or 'both'"

        self.df = pd.read_csv(csv_path)

        # drop text column rename column vector_10_binary_llm_text2 to text
        if use_words:
            # Rename column text 
            self.df = self.df.drop(columns=["text"])
            self.df = self.df.rename(columns={"vector_10_binary_llm_text2": "text"})        

        # Clean columns
        drop_cols = ['text_id', 'timestamp', 'collection_phase', 'vector_10_soft',
                     'state_change_val', 'state_change_aro', 'train', 'is_words']
        existing_cols = [col for col in drop_cols if col in self.df.columns]
        self.df = self.df.drop(columns=existing_cols)
        self.df = self.df.loc[:, ~self.df.columns.str.startswith('vector')]

        self.tokenizer = tokenizer
        self.window_size = window_size
        self.max_text_length = max_text_length

        # Group by user and sort
        self.user_sequences = {}
        for user_id, group in self.df.groupby("user_id"):
            group = group.sort_values("text_id_ordered")
            if train_dataset:
                group = group.iloc[:-1]
            self.user_sequences[user_id] = group.reset_index(drop=True)

        # Build instances (with partial windows)
        self.instances = []
        all_user_list = []

        for user_id, seq in self.user_sequences.items():
            # Need at least 2 entries: 1 for history + 1 for target
            if len(seq) < 2:
                continue

            # Use partial windows for short sequences
            min_history = min(window_size - 1, len(seq) - 1)

            for t in range(min_history, len(seq) - 1):
                self.instances.append((user_id, t))
                all_user_list.append(user_id)

        # Create user mapping
        unique_users = sorted(list(set(all_user_list)))
        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for u, i in self.user2idx.items()}

        print(f"Dataset: {len(self.instances)} instances from {len(self.user2idx)} users")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        user_id, t = self.instances[idx]
        seq = self.user_sequences[user_id]

        start = t - self.window_size + 1
        end = t + 1
        window = seq.iloc[start:end]

        # Texts
        texts = window["text"].tolist()
        tokenized = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors=None
        )

        # VA values
        va = torch.tensor(window[["valence", "arousal"]].values, dtype=torch.float)

        # Target delta based on predict_target
        v_t = seq.iloc[t]["valence"]
        a_t = seq.iloc[t]["arousal"]
        v_next = seq.iloc[t + 1]["valence"]
        a_next = seq.iloc[t + 1]["arousal"]

        if self.predict_target == 'valence':
            delta = torch.tensor([v_next - v_t], dtype=torch.float)
        elif self.predict_target == 'arousal':
            delta = torch.tensor([a_next - a_t], dtype=torch.float)
        else:  # 'both'
            delta = torch.tensor([v_next - v_t, a_next - a_t], dtype=torch.float)

        # Map user_id to index
        user_idx = self.user2idx[user_id]

        return {
            "user_id": user_idx,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "va_values": va,
            "delta": delta
        }


class StateChangeInferenceDataset(Dataset):
    """Inference dataset - one sample per user"""

    def __init__(self, csv_path: str, tokenizer, window_size: int,
                 user2idx: dict, unknown_id: int, max_text_length: int = 128,
                 predict_target: str = 'valence'):
        """
        Args:
            predict_target: 'valence', 'arousal', or 'both'
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.user2idx = user2idx
        self.unknown_id = unknown_id
        self.max_text_length = max_text_length
        self.predict_target = predict_target

        self.samples = []

        for user_id, user_df in self.df.groupby("user_id"):
            user_df = user_df.sort_values("text_id_ordered")

            # Map to training user idx, or unknown
            user_idx = self.user2idx.get(user_id, self.unknown_id)

            # Last entry is target point
            target_row = user_df.iloc[-1]
            idx = len(user_df) - 1

            # Get history (up to window_size previous entries)
            start_idx = max(0, idx - window_size)
            history_df = user_df.iloc[start_idx:idx]

            if len(history_df) == 0:
                # No history, use target row itself
                history_df = user_df.iloc[idx:idx+1]

            texts = history_df["text"].tolist()
            va_vals = history_df[["valence", "arousal"]].values.tolist()

            # Tokenize each text
            input_ids = []
            for txt in texts:
                enc = self.tokenizer(
                    txt,
                    truncation=True,
                    padding=False,
                    max_length=self.max_text_length,
                    return_attention_mask=False
                )
                input_ids.append(enc["input_ids"])

            # Dummy delta based on predict_target
            if self.predict_target == 'both':
                dummy_delta = torch.tensor([0.0, 0.0], dtype=torch.float)
            else:
                dummy_delta = torch.tensor([0.0], dtype=torch.float)

            self.samples.append({
                "user_id": user_idx,
                "input_ids": input_ids,
                "va_values": torch.tensor(va_vals, dtype=torch.float),
                "text_id_ordered": target_row["text_id_ordered"],
                "delta": dummy_delta
            })

        print(f"Inference dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ========================================
# COLLATE FUNCTION
# ========================================

def state_change_collate_fn(batch):
    """Pads sequences in batch"""

    user_id = torch.tensor([item["user_id"] for item in batch], dtype=torch.long)

    # VA values
    va_sequences = [item["va_values"] for item in batch]
    va_padded = pad_sequence(va_sequences, batch_first=True, padding_value=0.0)

    # Text tokens - nested structure
    nested_input_ids = []
    nested_attention_masks = []

    for item in batch:
        instance_seqs = []
        instance_masks = []
        for seq in item["input_ids"]:
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            instance_seqs.append(seq_tensor)
            mask = torch.ones_like(seq_tensor)
            instance_masks.append(mask)

        nested_input_ids.append(instance_seqs)
        nested_attention_masks.append(instance_masks)

    # Pad within each instance
    padded_input_instances = []
    padded_mask_instances = []

    for seqs, masks in zip(nested_input_ids, nested_attention_masks):
        padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
        padded_masks = pad_sequence(masks, batch_first=True, padding_value=0)
        padded_input_instances.append(padded_seqs)
        padded_mask_instances.append(padded_masks)

    # Pad across batch
    max_num_seqs = max(p.size(0) for p in padded_input_instances)
    max_seq_len = max(p.size(1) for p in padded_input_instances)

    batch_size = len(batch)
    input_ids_batch = torch.zeros(batch_size, max_num_seqs, max_seq_len, dtype=torch.long)
    attention_mask_batch = torch.zeros(batch_size, max_num_seqs, max_seq_len, dtype=torch.long)

    for i, (input_padded, mask_padded) in enumerate(zip(padded_input_instances, padded_mask_instances)):
        input_ids_batch[i, :input_padded.size(0), :input_padded.size(1)] = input_padded
        attention_mask_batch[i, :mask_padded.size(0), :mask_padded.size(1)] = mask_padded

    # Targets
    delta = torch.stack([item["delta"] for item in batch])

    return {
        "user_id": user_id,
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "va_values": va_padded,
        "delta": delta
    }


# ========================================
# TEXT ENCODER
# ========================================

class TextEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        # Mean pooling
        emb = last_hidden.mean(dim=1)
        return emb


# ========================================
# IMPROVED MODEL
# ========================================

class SimpleStateChangeModel(nn.Module):
    """Simplified model that works better when window_size=1"""

    def __init__(self, num_users, text_emb_dim, hidden_dim=64, user_emb_dim=16,
                 predict_target='valence', use_text=True):
        super().__init__()

        self.predict_target = predict_target
        self.use_text = use_text

        # Smaller user embedding
        self.user_embedding = nn.Embedding(num_users + 1, user_emb_dim)
        self.unknown_user_id = num_users

        # Input: text (optional) + current VA + previous delta
        input_dim = (text_emb_dim if use_text else 0) + 2 + 2

        # Much simpler architecture
        
        if predict_target == 'both':
            output_dim = 2
        else:
            output_dim = 1

        self.regressor = nn.Sequential(
            nn.Linear(input_dim + user_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, text_embeddings, va_values, user_id):
        # Map unseen users
        user_id = torch.clamp(user_id, 0, self.unknown_user_id)
        user_emb = self.user_embedding(user_id)

        # Use only CURRENT timestep
        if self.use_text:
            current_text = text_embeddings[:, -1, :]  # (B, D)
        current_va = va_values[:, -1, :]              # (B, 2)

        # Compute previous delta (momentum feature)
        if va_values.size(1) > 1:
            prev_delta = va_values[:, -1:, :] - va_values[:, -2:-1, :]
            prev_delta = prev_delta.squeeze(1)  # (B, 2)
        else:
            prev_delta = torch.zeros_like(current_va)

        # Concatenate features
        if self.use_text:
            features = torch.cat([current_text, current_va, prev_delta, user_emb], dim=-1)
        else:
            features = torch.cat([current_va, prev_delta, user_emb], dim=-1)

        # Predict delta directly (no residual - let model learn it)
        delta = self.regressor(features)

        return delta


# ========================================
# TRAINING
# ========================================

def train_model(train_path, window_size=1, batch_size=32, num_epochs=30,  # Changed default to 1
                hidden_dim=64, user_emb_dim=16, lr=5e-4, predict_target='valence',
                use_text=True, model_name='sentence-transformers/all-MiniLM-L6-v2', use_words=True, train_dataset: bool = True):  # New parameter
    """
    Args:
        predict_target: 'valence', 'arousal', or 'both'
        use_text: Whether to use text embeddings (set False to test no-text baseline)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Prediction target: {predict_target}")
    print(f"Window size: {window_size}")
    print(f"Using text: {use_text}")

    # Load tokenizer and encoder
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder_model = AutoModel.from_pretrained(model_name)
    text_encoder_model.eval()
    for param in text_encoder_model.parameters():
        param.requires_grad = False

    text_encoder = TextEncoderWrapper(text_encoder_model).to(device)

    # Load dataset
    dataset = StateChangeDataset(
        csv_path=train_path,
        tokenizer=tokenizer,
        window_size=window_size,
        predict_target=predict_target,
        use_words=use_words,
        train_dataset=train_dataset,
    )

    # Train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=state_change_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=state_change_collate_fn
    )

    # Initialize model
    num_users = len(dataset.user2idx)
    D_text = 768  # MiniLM dimension

    model = SimpleStateChangeModel(
        num_users=num_users,
        text_emb_dim=D_text,
        hidden_dim=hidden_dim,
        user_emb_dim=user_emb_dim,
        predict_target=predict_target,
        use_text=use_text
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_bar:
            user_id = batch['user_id'].to(device)
            va_values = batch['va_values'].to(device)
            delta_true = batch['delta'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            B, W, L = input_ids.shape

            # Encode text (only if using text)
            if use_text:
                with torch.no_grad():
                    text_emb = text_encoder(
                        input_ids.view(B*W, L),
                        attention_mask.view(B*W, L)
                    ).view(B, W, -1)
            else:
                text_emb = torch.zeros(B, W, 1).to(device)  # Dummy

            # Forward
            delta_pred = model(text_emb, va_values, user_id)
            loss = criterion(delta_pred, delta_true)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * B
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_bar:
                user_id = batch['user_id'].to(device)
                va_values = batch['va_values'].to(device)
                delta_true = batch['delta'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                B, W, L = input_ids.shape

                text_emb = text_encoder(
                    input_ids.view(B*W, L),
                    attention_mask.view(B*W, L)
                ).view(B, W, -1)

                delta_pred = model(text_emb, va_values, user_id)
                loss = criterion(delta_pred, delta_true)
                val_loss += loss.item() * B
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= len(val_dataset)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'user2idx': dataset.user2idx,
                'num_users': num_users,
                'window_size': window_size,
                'hidden_dim': hidden_dim,
                'user_emb_dim': user_emb_dim,
                'predict_target': predict_target,
                'use_text': use_text,
                'val_loss': val_loss
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return checkpoint


# ========================================
# INFERENCE
# ========================================

def run_inference(test_path, checkpoint_path, batch_size=32, output_path='predictions.csv', model_name = 'sentence-transformers/all-MiniLM-L6-v2', train_dataset = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    user2idx = checkpoint['user2idx']
    num_users = checkpoint['num_users']
    window_size = checkpoint['window_size']
    hidden_dim = checkpoint['hidden_dim']
    user_emb_dim = checkpoint['user_emb_dim']
    predict_target = checkpoint.get('predict_target', 'valence')
    use_text = checkpoint.get('use_text', True)

    print(f"Loaded checkpoint with {num_users} training users")
    print(f"Prediction target: {predict_target}")
    print(f"Using text: {use_text}")

    # Load tokenizer and encoder
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_encoder_model = AutoModel.from_pretrained(model_name)
    text_encoder_model.eval()
    for param in text_encoder_model.parameters():
        param.requires_grad = False

    text_encoder = TextEncoderWrapper(text_encoder_model).to(device)

    # Load model
    D_text = 768  # MiniLM dimension
    model = SimpleStateChangeModel(
        num_users=num_users,
        text_emb_dim=D_text,
        hidden_dim=hidden_dim,
        user_emb_dim=user_emb_dim,
        predict_target=predict_target,
        use_text=use_text
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test dataset
    unknown_id = num_users  # Points to the unknown user embedding
    test_dataset = StateChangeInferenceDataset(
        csv_path=test_path,
        tokenizer=tokenizer,
        window_size=window_size,
        user2idx=user2idx,
        unknown_id=unknown_id,
        predict_target=predict_target
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=state_change_collate_fn
    )

    # Run inference
    all_predictions = []
    all_text_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            va_values = batch["va_values"].to(device)
            user_id = batch["user_id"].to(device)

            B, S, T = input_ids.shape

            # Encode text (only if model uses text)
            if use_text:
                input_ids_flat = input_ids.view(B * S, T)
                attention_mask_flat = attention_mask.view(B * S, T)

                text_embeddings = text_encoder(
                    input_ids=input_ids_flat,
                    attention_mask=attention_mask_flat
                )
                text_embeddings = text_embeddings.view(B, S, -1)
            else:
                text_embeddings = torch.zeros(B, S, 1).to(device)

            # Model forward
            delta_pred = model(
                text_embeddings=text_embeddings,
                va_values=va_values,
                user_id=user_id
            )

            # Collect predictions
            for d in delta_pred.cpu():
                if predict_target == 'valence':
                    all_predictions.append({"valence_delta": d.item()})
                elif predict_target == 'arousal':
                    all_predictions.append({"arousal_delta": d.item()})
                else:  # 'both'
                    all_predictions.append({
                        "valence_delta": d[0].item(),
                        "arousal_delta": d[1].item()
                    })

    # Create output DataFrame
    pred_df = pd.DataFrame(all_predictions)

    if train_dataset:
        pred_list_user_id = [1, 2, 3, 4, 6, 7, 8, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 98, 99, 103, 105, 106, 107, 108, 109, 113, 114, 116, 119, 120, 121, 122, 124, 125, 126, 127, 128, 131, 133, 134, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 150, 152, 153, 155, 158, 161, 162, 163, 164, 166, 167, 168, 169, 170, 171, 172, 173, 174, 176, 178, 179, 180, 182]
    else:
        pred_list_user_id = [6, 8, 11, 16, 21, 27, 29, 30, 38, 41, 46, 47, 50, 51, 56, 59, 66, 68, 74, 76, 78, 86, 88, 90,
                            93, 95, 96, 98, 109, 113, 114, 116, 121, 128, 137, 142, 144, 146, 148, 153, 161, 162, 167, 176,
                            178, 182]

    # pred_df.to_csv('pred1.csv', index=False)

    # Assign correct column names to existing predictions
    if predict_target == 'both':
        pred_df.columns = [
            "pred_state_change_valence",
            "pred_state_change_arousal"
        ]
    elif predict_target == 'arousal':
        pred_df.columns = [
            "pred_state_change_arousal",
        ]
        # add a dummy valence column with zeros
        pred_df.insert(0, "pred_state_change_valence", [0.0]*len(pred_df))
    else:  # 'valence'
        pred_df.columns = [
            "pred_state_change_valence",
        ]
        # add a dummy arousal column with zeros
        pred_df.insert(1, "pred_state_change_arousal", [0.0]*len(pred_df))
    # Insert user_id as the first column
    pred_df.insert(0, "user_id", pred_list_user_id)
    # Save to CSV
    # pred_df.to_csv("pred90.csv", index=False)

    # Add text_id_ordered from original test file
    # Save
    pred_df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to {output_path}")
    print(pred_df.head(10))

    return pred_df


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    # Configuration
    PREDICT_TARGET = 'both'  # 'valence', 'arousal', or 'both'
    WINDOW_SIZE = 4  # Start with 1 based on your findings
    USE_TEXT = False  # Set to False to test no-text baseline
    USER_EMB_DIM = 8  # Smaller user embedding dimension
    USE_WORDS = False  # Set to False to test no-words baseline
    MODEL_NAME = 'FacebookAI/roberta-base'

    TRAIN_PATH = "train_data_padded.csv"
    TEST_PATH = "./semeval_test_subtask2_5padded.csv"
    TRAIN_DATASET_FLAG = False

    # Training
    print("=" * 50)
    print("TRAINING")
    print("=" * 50)

    checkpoint = train_model(
        train_path=TRAIN_PATH,
        window_size=WINDOW_SIZE,
        train_dataset=TRAIN_DATASET_FLAG,
        batch_size=32,
        num_epochs=10,
        hidden_dim=128,
        user_emb_dim=USER_EMB_DIM,
        lr=5e-4,
        predict_target=PREDICT_TARGET,
        use_text=USE_TEXT,
        model_name=MODEL_NAME,
        use_words=USE_WORDS
    )
    #  other models to try which are good at sentiment analysis:
    # 'distilbert-base-uncased-finetuned-sst-2-english'
    # 'roberta-base-finetuned-sentiment'

    # Inference
    print("\n" + "=" * 50)
    print("INFERENCE")
    print("=" * 50)

    if USE_TEXT:
        if USE_WORDS:
            output_suffix = 'withwords'
        else:
            output_suffix = 'withtext'
    else:
        output_suffix = 'notext'

    model_for_file = MODEL_NAME.replace('/', '_')

    predictions = run_inference(
        test_path=TEST_PATH,
        train_dataset=TRAIN_DATASET_FLAG,
        checkpoint_path="best_model_for_arousal.pth",
        batch_size=32,
        output_path=f'predictions_window_{WINDOW_SIZE}_{model_for_file}_user{USER_EMB_DIM}_{PREDICT_TARGET}_{output_suffix}_all.csv',
        model_name='FacebookAI/roberta-base'
    )
    
    