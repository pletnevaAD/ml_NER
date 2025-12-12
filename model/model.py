import json

import torch
from torch import nn
from torchcrf import CRF

from model.consts import DEVICE, LR, WEIGHT_DECAY

DROPOUT_EMB = 0.5
DROPOUT_LSTM = 0.5
DROPOUT_FC = 0.33

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

    def loss(self, x, tags_list, mask):
        emissions = self.forward(x, mask)
        return -self.crf(emissions, tags_list, mask=mask, reduction='mean')

    def predict(self, x, mask):
        emissions = self.forward(x, mask)
        return self.crf.decode(emissions, mask=mask)


with open("word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)
vocab_size = len(word2idx)
NER_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
model = BiLSTM_CRF(vocab_size, len(NER_TAGS)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
