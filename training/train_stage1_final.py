import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= CONFIG =================
FEATURE_DIR = "stage1_features"
AI_DIR = os.path.join(FEATURE_DIR, "ai")
HUMAN_DIR = os.path.join(FEATURE_DIR, "human")

BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3
MODEL_OUT = "stage1_detector.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= DATASET =================
class FeatureDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for label, sub in [(1, "ai"), (0, "human")]:
            folder = os.path.join(root, sub)
            for f in os.listdir(folder):
                if f.endswith(".pt"):
                    self.samples.append(
                        (os.path.join(folder, f), label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError(" Dataset is empty")

        print(f"Loaded {len(self.samples)} feature files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        feat = torch.load(path, map_location="cpu")
        if isinstance(feat, dict):
            feat = feat["embedding"]

        feat = feat.float()
        label = torch.tensor(label, dtype=torch.float32)

        return feat, label

# ================= MODEL =================
class Stage1Classifier(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LayerNorm(256),   # ✅ SAFE normalization
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

# ================= TRAIN =================
def main():
    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = FeatureDataset(FEATURE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = Stage1Classifier().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).unsqueeze(1)  # ✅ shape [B,1]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Loss: {avg:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Stage-1 model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
