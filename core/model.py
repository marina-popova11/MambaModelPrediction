import torch.nn as nn
import torch.optim as optim

class MambaModel(nn.Module):
    def __init__(self, input_dim, n_classes=5,  d_model=128, n_layers=3, use_mamba=True):
        super().__init__()
        self.use_mamba = use_mamba
        self.n_classes = n_classes
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.input_proj = nn.Linear(input_dim, d_model)

        if use_mamba:
            try:
                from mamba_ssm import Mamba
                print("Using Mamba Architecture")
                self.blocks = nn.ModuleList([
                    Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                    for _ in range(n_layers)
                ])
            except ImportError:
                print("Mamba not installed. Falling back to LSTM.")
                self.use_mamba = False

        if not self.use_mamba:
            print("Using LSTM Architecture (Fallback)")
            self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, batch_first=True, dropout=0.0)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.drop1(x)
        if self.use_mamba:
            for block in self.blocks:
                x = block(x)
        else:
            x, _ = self.lstm(x)
        x = self.drop2(x)
        x = self.norm(x)
        last_hidden = x[:, -1, :]
        logits = self.head(last_hidden)
        return logits