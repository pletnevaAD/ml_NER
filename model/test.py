import os

import torch
from datasets import load_dataset
from seqeval.metrics import f1_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from consts import MODEL_DIR, DEVICE, BATCH_SIZE
from model import model
from model.get_dataset import CoNLLDataset, dataset


def test():
    test_loader = DataLoader(CoNLLDataset(dataset['test']), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

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
