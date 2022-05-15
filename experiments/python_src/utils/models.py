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


class HeatModelNoBias(nn.Module):
    def __init__(self, n):
        super(HeatModelNoBias, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n, n, bias=False),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, t, u):
        return self.net(u)


class BurgersModelA(nn.Module):
    def __init__(self, n):
        super(BurgersModelA, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n, out_features=int(n/2)),
            nn.Tanh(),
            nn.Linear(in_features=int(n/2), out_features=int(n/2)),
            nn.Tanh(),
            nn.Linear(in_features=int(n/2), out_features=int(n/2)),
            nn.Tanh(),
            nn.Linear(in_features=int(n/2), out_features=n),
            nn.Tanh()
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, u):
        return self.net(u)


class BurgersModelB(nn.Module):
    def __init__(self, n):
        super(BurgersModelB, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, n, n, 3),
            nn.ELU(),
            nn.Conv1d(1, n, int(n/2), 3),
            nn.ELU(),
            nn.ConvTranspose1d(1, int(n/2), n, 3),
            nn.ELU(),
            nn.ConvTranspose1d(1, n, n, 3),
            nn.Tanh()
        )
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, u):
        return self.net(u)
