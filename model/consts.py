import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = "checkpoints_best"
BATCH_SIZE = 32
MAX_EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 7
EMBED_DIM = 300
HIDDEN_DIM = 256
PATIENCE = 10
