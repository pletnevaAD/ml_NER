import os

import torch
from seqeval.metrics import f1_score, classification_report
from tqdm import tqdm

from model.consts import MODEL_DIR, DEVICE
from model.get_dataset import get_dataset
from model.model import model


def test():
    NER_TAGS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    tag2idx = {t: i for i, t in enumerate(NER_TAGS)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    _, _, test_loader = get_dataset()

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for x, y_list, mask in tqdm(test_loader, desc="Test"):
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            preds = model.predict(x, mask)
            for p, t, m in zip(preds, y_list, mask):
                length = m.sum().item()
                all_trues.append([idx2tag[i] for i in t[:length].cpu().tolist()])
                all_preds.append([idx2tag[i] for i in p])

    test_f1 = f1_score(all_trues, all_preds)
    print(f"FINAL TEST F1 (seqeval): {test_f1:.4f}")
    print(classification_report(all_trues, all_preds, digits=4))
