# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset
import logging
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from seqeval.metrics import classification_report, f1_score

# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# -----------------------------
BATCH_SIZE = 16
MAX_EPOCHS = 50
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_HEADS = 8
LR = 5e-4
CLIP_VALUE = 1.0
PATIENCE = 7
MIN_DELTA = 0.001
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
dataset = load_dataset("lhoestq/conll2003")

NER_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
tag2idx = {tag: idx for idx, tag in enumerate(NER_TAGS)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}
num_tags = len(NER_TAGS)

# -----------------------------
def build_vocab(data, min_freq=2):
    counter = Counter()
    for example in data:
        counter.update([t.lower() for t in example['tokens']])
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

word2idx = build_vocab(dataset['train'], min_freq=2)
vocab_size = len(word2idx)
print(f"Vocab size: {vocab_size} | Tags: {num_tags}")

# -----------------------------
class CoNLLDataset(Dataset):
    def __init__(self, split):
        self.data = split
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        return self.data[i]['tokens'], self.data[i]['ner_tags']

# -----------------------------
def collate_fn(batch):
    tokens_list, labels_list = zip(*batch)
    pad_idx = word2idx['<pad>']
    unk_idx = word2idx['<unk>']

    # Токены → индексы
    seqs = [[word2idx.get(t.lower(), unk_idx) for t in tokens] for tokens in tokens_list]
    x = [torch.tensor(s, dtype=torch.long) for s in seqs]
    y = [torch.tensor(l, dtype=torch.long) for l in labels_list]

    x_padded = pad_sequence(x, batch_first=True, padding_value=pad_idx)
    # ВАЖНО: заменяем -100 на 0 ('O') перед подачей в CRF!
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)  # не -100!
    y_padded[y_padded == -100] = 0  # на всякий случай, если где-то осталось

    mask = (x_padded != pad_idx)  # bool mask: True = реальный токен

    return x_padded, y_padded, mask

# -----------------------------
train_loader = DataLoader(CoNLLDataset(dataset['train']), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(CoNLLDataset(dataset['validation']), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(CoNLLDataset(dataset['test']), batch_size=1, shuffle=False, collate_fn=collate_fn)

class BiLSTM_Transformer_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim=128, hidden_dim=256, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['<pad>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=512,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, mask):
        e = self.embedding(x)
        lstm_out, _ = self.lstm(e)
        trans_out = self.transformer(lstm_out, src_key_padding_mask=~mask)
        emissions = self.fc(self.dropout(trans_out))
        return emissions

    def loss(self, x, tags, mask):
        emissions = self.forward(x, mask)
        # mask должен быть uint8 для TorchCRF
        return -self.crf(emissions, tags, mask=mask.to(torch.uint8))

    def predict(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask.to(torch.uint8))

# -----------------------------
model = BiLSTM_Transformer_CRF(vocab_size, num_tags, EMBED_DIM, HIDDEN_DIM, NUM_HEADS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter('runs/conll_final')

best_val_loss = float('inf')
patience_counter = 0

logger.info("=== START TRAINING ===")

for epoch in range(1, MAX_EPOCHS + 1):
    # === TRAIN ===
    model.train()
    train_loss = 0.0
    for x, y, mask in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        loss = model.loss(x, y, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # === VALIDATION ===
    model.eval()
    val_loss = 0.0
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x, y, mask in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            loss = model.loss(x, y, mask)
            val_loss += loss.item()

            preds = model.predict(x, mask)
            for i in range(x.size(0)):
                length = mask[i].sum().item()
                true = [idx2tag[t] for t in y[i][:length].cpu().tolist()]
                pred = [idx2tag[p] for p in preds[i]]
                all_trues.append(true)
                all_preds.append(pred)

    val_loss /= len(val_loader)
    val_f1 = f1_score(all_trues, all_preds)

    logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Metric/F1_val", val_f1, epoch)

    # === Сохранение ===
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_f1': val_f1,
        'word2idx': word2idx,
        'idx2tag': idx2tag
    }, os.path.join(MODEL_DIR, f"model_epoch_{epoch}.pth"))
    logger.info(f"Сохранена модель: model_epoch_{epoch}.pth")

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
        logger.info("Новая лучшая модель!")
    else:
        patience_counter += 1
        logger.info(f"Нет улучшения {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            logger.info("Early stopping!")
            break

writer.close()

# === ТЕСТ ===
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for x, y, mask in tqdm(test_loader, desc="Test"):
        x = x.to(DEVICE)
        preds = model.predict(x, mask)[0]
        length = mask[0].sum().item()
        true = [idx2tag[t] for t in y[0][:length].cpu().tolist()]
        pred = [idx2tag[p] for p in preds]
        all_trues.append(true)
        all_preds.append(pred)

test_f1 = f1_score(all_trues, all_preds)
print("\n" + "="*60)
print(f"FINAL TEST F1: {test_f1:.4f}")
print(classification_report(all_trues, all_preds))
print("="*60)