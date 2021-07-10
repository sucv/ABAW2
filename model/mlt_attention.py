from torch import nn

class Attn_Net_Gated(nn.Module):
    def __init__(self, indim=1024, hiddim=256, dropout=False, n_tasks=2):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(indim, hiddim),
            nn.Tanh()
        ]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*self.attention_a)
        if n_tasks >= 2:
            self.attention_b = [
                nn.Linear(indim, hiddim * (n_tasks - 1)),
                nn.Sigmoid()
            ]
            if dropout:
                self.attention_b.append(nn.Dropout(0.25))
            self.attention_b = nn.Sequential(*self.attention_b)
        self.hiddim = hiddim
        self.n_tasks = n_tasks
        self.out = nn.Linear(hiddim, n_tasks)

    def forward(self, x):
        B, C = x.shape
        a = self.attention_a(x)
        if self.n_tasks >= 2:
            b = self.attention_b(x).view(B, self.hiddim, self.n_tasks - 1)
            for i in range(self.n_tasks - 1):
                a = a.mul(b[..., i])
        a = self.out(a)  # N x n_classes
        return a
