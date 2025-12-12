import logging
import os

import torch
from datasets import load_dataset
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from consts import DEVICE, MAX_EPOCHS, MODEL_DIR, PATIENCE, BATCH_SIZE
from get_dataset import idx2tag, build_vocab, CoNLLDataset, collate_fn, dataset
from model import BiLSTM_CRF, model, optimizer, scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    logger.info(f"Using device: {DEVICE}")
    train_loader = DataLoader(CoNLLDataset(dataset['train']), batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(CoNLLDataset(dataset['validation']), batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn)

    best_f1 = 0.0
    patience_cnt = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x, y_list, mask in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            x = x.to(DEVICE)
            mask = mask.to(DEVICE)
            optimizer.zero_grad()
            loss = model.loss(x, y_list, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        all_preds, all_trues = [], []
        val_loss = 0.0
        with torch.no_grad():
            for x, y_list, mask in val_loader:
                x = x.to(DEVICE)
                mask = mask.to(DEVICE)
                loss = model.loss(x, y_list, mask)
                preds = model.predict(x, mask)
                for p, t, m in zip(preds, y_list, mask):
                    length = m.sum().item()
                    all_trues.append([idx2tag[i] for i in t[:length].cpu().tolist()])
                    all_preds.append([idx2tag[i] for i in p])

        val_loss /= len(val_loader)
        val_f1 = f1_score(all_trues, all_preds)

        logger.info(
            f"Epoch {epoch:02d} | Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1 + 0.001:
            best_f1 = val_f1
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            logger.info("New best model saved!")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                logger.info("Early stopping")
                break
