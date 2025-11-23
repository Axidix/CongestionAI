import torch.nn as nn
import torch

class MLPForecaster(nn.Module):
    def __init__(self, input_length, num_features, horizon, num_detectors, emb_dim=16):
        super().__init__()

        self.emb = nn.Embedding(num_detectors, emb_dim)

        # after concatenation embedding is repeated for each timestep → shape (input_length, emb_dim)
        self.input_dim = input_length * (num_features + emb_dim)

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, horizon)
        )

    def forward(self, x, det_ids):
        """
        x: (B, T, F)
        det_ids: (B,)
        """
        emb = self.emb(det_ids)          # (B, emb_dim)
        emb_rep = emb.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, T, emb_dim)

        x = torch.cat([x, emb_rep], dim=-1)  # (B, T, F+emb_dim)
        x = x.reshape(x.size(0), -1)         # flatten
        return self.net(x)