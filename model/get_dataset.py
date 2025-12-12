import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from collections import Counter


def build_vocab(data, min_freq=1):
    counter = Counter()
    for ex in data:
        counter.update([w.lower() for w in ex['tokens']])
    vocab = {'<pad>': 0, '<unk>': 1}
    for w, f in counter.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab

dataset = load_dataset("lhoestq/conll2003")

NER_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
tag2idx = {t: i for i, t in enumerate(NER_TAGS)}
idx2tag = {i: t for t, i in tag2idx.items()}

word2idx = build_vocab(dataset['train'], min_freq=1)
vocab_size = len(word2idx)
print(f"Vocab size: {vocab_size}")

def collate_fn(batch):
    tokens, tags = zip(*batch)
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
