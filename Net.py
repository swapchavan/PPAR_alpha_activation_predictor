import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features, n_hid_lay=3, neurons=64, dropout=0.2, n_classes=2):
        super().__init__()
        self.name = "DNN"

        layers = [
            nn.Linear(n_features, neurons),
            #nn.BatchNorm1d(neurons),
            nn.LayerNorm(neurons),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_hid_lay):
            layers += [
                nn.Linear(neurons, neurons),
                #nn.BatchNorm1d(neurons),
                nn.LayerNorm(neurons),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]

        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(neurons, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)  # logits [B, n_classes]

    @torch.no_grad()
    def predict_prob(self, logits):
        # logits: [B, 2]
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_class(self, logits):
        # logits: [B, 2]
        return logits.argmax(dim=1)
