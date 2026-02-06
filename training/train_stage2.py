import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.dataset_stage2 import Stage2Dataset
from stage2_model import Stage2Verifier

DATA_DIR = "stage_2\stage2_data"
BATCH = 4
EPOCHS = 5
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("Using device:", DEVICE)

    ds = Stage2Dataset(DATA_DIR)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = Stage2Verifier().to(DEVICE)

    #  ASYMMETRIC LOSS
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([2.5]).to(DEVICE)
    )

    opt = torch.optim.AdamW(model.head.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total = 0

        for x, y in tqdm(dl, desc=f"Epoch {epoch+1}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1} loss: {total/len(dl):.4f}")
        torch.save(model.state_dict(), f"stage2_verifier_epoch{epoch+1}.pt")

    print(" Stage-2 training complete")

if __name__ == "__main__":
    main()