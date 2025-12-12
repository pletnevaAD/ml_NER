import json

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from model.consts import BATCH_SIZE


def build_vocab(data, min_freq=1):
    counter = Counter()
    for ex in data:
        counter.update([w.lower() for w in ex['tokens']])
    vocab = {'<pad>': 0, '<unk>': 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab


def get_dataset():
    dataset = load_dataset("lhoestq/conll2003")

    NER_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    tag2idx = {t: i for i, t in enumerate(NER_TAGS)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    word2idx = build_vocab(dataset['train'], min_freq=1)

    with open("word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(CoNLLDataset(dataset['train']), batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(CoNLLDataset(dataset['validation']), batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(CoNLLDataset(dataset['test']), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def collate_fn(batch):
    tokens, tags = zip(*batch)

    with open("word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    pad_idx = word2idx['<pad>']
    unk_idx = word2idx['<unk>']

    seqs = [[word2idx.get(t.lower(), unk_idx) for t in sent] for sent in tokens]
    x = [torch.tensor(s, dtype=torch.long) for s in seqs]
    y = [torch.tensor(t, dtype=torch.long) for t in tags]

    x_pad = pad_sequence(x, batch_first=True, padding_value=pad_idx)
    y_pad = pad_sequence(y, batch_first=True, padding_value=-100)
    mask = (x_pad != pad_idx)

    return x_pad, y_pad, mask


class CoNLLDataset(Dataset):
    def __init__(self, split): self.data = split

    def __len__(self): return len(self.data)

    def __getitem__(self, i): return self.data[i]['tokens'], self.data[i]['ner_tags']
