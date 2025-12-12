import json
import re

import torch

from model.consts import DEVICE
from model.model import model


def predict_sentence(sentence: str):
    sentence = sentence.lower()
    with open("word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    NER_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    tag2idx = {t: i for i, t in enumerate(NER_TAGS)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    tokens = re.findall(r"\w+(?:'\w+)?|[.,!?;\"'()\[\]]", sentence)
    token_ids = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
    x = torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

    mask = (x != word2idx["<pad>"]).to(DEVICE)

    with torch.no_grad():
        emissions = model.forward(x, mask)
        prediction = model.crf.decode(emissions, mask=mask.to(torch.uint8))[0]

    tags = [idx2tag[idx] for idx in prediction]

    return [{"word": w, "tag": t} for w, t in zip(tokens, tags)]
