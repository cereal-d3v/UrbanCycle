# hazard_prediction.py
import torch
import torch.nn as nn
import numpy as np

class HazardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(HazardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50)
        c_0 = torch.zeros(2, x.size(0), 50)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# Train the model with historical hazard data
