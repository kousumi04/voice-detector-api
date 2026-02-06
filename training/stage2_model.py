import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Stage2Verifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # FULLY FREEZE ENCODER
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x).last_hidden_state.mean(dim=1)
        return self.head(h).squeeze(1)