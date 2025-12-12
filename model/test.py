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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ======================= HYPERPARAMS =======================
BATCH_SIZE = 32
MAX_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 6
EMBED_DIM = 300          # лучше 300, даже если случайные
HIDDEN_DIM = 256         # BiLSTM hidden (будет 512 после bidirectional)
DROPOUT_EMB = 0.5
DROPOUT_LSTM = 0.5
DROPOUT_FC = 0.33
PATIENCE = 10
MODEL_DIR = "checkpoints_best"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================= DATA =======================
dataset = load_dataset("lhoestq/conll2003")  # правильное имя, а не lhoestq/conll2003

NER_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
tag2idx = {t: i for i, t in enumerate(NER_TAGS)}
idx2tag = {i: t for t, i in tag2idx.items()}

def build_vocab(data, min_freq=1):  # min_freq=1 даёт лучше результат
    counter = Counter()
    for ex in data:
        counter.update([w.lower() for w in ex['tokens']])
    vocab = {'<pad>': 0, '<unk>': 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab

word2idx = build_vocab(dataset['train'], min_freq=1)
vocab_size = len(word2idx)
print(f"Vocab size: {vocab_size}")

class CoNLLDataset(Dataset):
    def __init__(self, split): self.data = split
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]['tokens'], self.data[i]['ner_tags']

def collate_fn(batch):
    tokens, tags = zip(*batch)
    pad_idx = word2idx['<pad>']
    unk_idx = word2idx['<unk>']

    seqs = [[word2idx.get(t.lower(), unk_idx) for t in sent] for sent in tokens]
    x = [torch.tensor(s, dtype=torch.long) for s in seqs]
    y = [torch.tensor(t, dtype=torch.long) for t in tags]

    x_pad = pad_sequence(x, batch_first=True, padding_value=pad_idx)
    y_pad = pad_sequence(y, batch_first=True, padding_value=0)  # важно для CRF!
    mask = (x_pad != pad_idx)  # bool mask

    return x_pad, y_pad, mask

train_loader = DataLoader(CoNLLDataset(dataset['train']), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(CoNLLDataset(dataset['validation']), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(CoNLLDataset(dataset['test']), batch_size=32, shuffle=False, collate_fn=collate_fn)

# ======================= MODEL =======================
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embed_dim=300, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(DROPOUT_EMB)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True,
                            dropout=DROPOUT_LSTM if 2 > 1 else 0)

        self.dropout = nn.Dropout(DROPOUT_FC)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, mask):
        emb = self.embed_dropout(self.embed(x))
        lstm_out, _ = self.lstm(emb)
        emissions = self.fc(self.dropout(lstm_out))
        return emissions

    def loss(self, x, tags, mask):
        emissions = self.forward(x, mask)
        return -self.crf(emissions, tags, mask=mask, reduction='mean')

    def predict(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask)

model = BiLSTM_CRF(vocab_size, len(NER_TAGS)).to(DEVICE)
print(model)

# optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#
# # Cosine scheduler с warmup
# total_steps = len(train_loader) * MAX_EPOCHS
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=LR,
#     total_steps=total_steps,
#     pct_start=WARMUP_EPOCHS/MAX_EPOCHS,
#     anneal_strategy='cos',
#     div_factor=25,
#     final_div_factor=1e4
# )
#
# # ======================= TRAINING LOOP =======================
# best_f1 = 0.0
# patience_cnt = 0
#
# for epoch in range(1, MAX_EPOCHS + 1):
#     model.train()
#     train_loss = 0.0
#     for x, y, mask in tqdm(train_loader, desc=f"Epoch {epoch} train"):
#         x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
#         optimizer.zero_grad()
#         loss = model.loss(x, y, mask)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#         optimizer.step()
#         scheduler.step()
#         train_loss += loss.item()
#     train_loss /= len(train_loader)
#
#     # Validation
#     model.eval()
#     all_preds, all_trues = [], []
#     val_loss = 0.0
#     with torch.no_grad():
#         for x, y, mask in val_loader:
#             x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
#             loss = model.loss(x, y, mask)
#             val_loss += loss.item()
#             preds = model.predict(x, mask)
#             for p, t, m in zip(preds, y, mask):
#                 length = m.sum().item()
#                 all_trues.append([idx2tag[i] for i in t[:length].cpu().tolist()])
#                 all_preds.append([idx2tag[i] for i in p])
#
#     val_loss /= len(val_loader)
#     val_f1 = f1_score(all_trues, all_preds)
#
#     logger.info(f"Epoch {epoch:02d} | Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | Val F1: {val_f1:.4f}")
#
#     if val_f1 > best_f1 + 0.001:
#         best_f1 = val_f1
#         patience_cnt = 0
#         torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
#         logger.info("New best model saved!")
#     else:
#         patience_cnt += 1
#         if patience_cnt >= PATIENCE:
#             logger.info("Early stopping")
#             break
#
# # ======================= TEST =======================
# model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
# model.eval()
# all_preds, all_trues = [], []
#
# with torch.no_grad():
#     for x, y, mask in tqdm(test_loader, desc="Test"):
#         x, mask = x.to(DEVICE), mask.to(DEVICE)
#         preds = model.predict(x, mask)
#         for p, t, m in zip(preds, y, mask):
#             length = m.sum().item()
#             all_trues.append([idx2tag[i] for i in t[:length].cpu().tolist()])
#             all_preds.append([idx2tag[i] for i in p])
#
# test_f1 = f1_score(all_trues, all_preds)
# print("\n" + "="*60)
# print(f"FINAL TEST F1 (seqeval): {test_f1:.4f}")
# print(classification_report(all_trues, all_preds, digits=4))
# print("="*60)