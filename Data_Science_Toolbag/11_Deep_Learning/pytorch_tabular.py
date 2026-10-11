import torch
import torch.nn as nn

class TabularNN(nn.Module):
    """Kaggle ve dev verisetleri için donanım hızlandırmalı Yapay Sinir Ağı."""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Sınıflandırma için sona nn.Sigmoid eklenebilir
        )
        
    def forward(self, x):
        return self.network(x)

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
