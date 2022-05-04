import torch.nn as nn


class HeatModel(nn.Module):
    def __init__(self, n):
        super(HeatModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, n),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, u):
        return self.net(u)
