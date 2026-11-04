"""Models of the native pipeline and the linear head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):                      # TAB: 16-256-256-26, ReLU, no dropout
    def __init__(self, d_in=16, K=26, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K))

    def forward(self, x):
        return self.net(x)


def resnet18_cifar(K=10):                  # IMG: 3x3 stride-1 stem, no max-pool
    from torchvision.models import resnet18
    m = resnet18(num_classes=K)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


class FCN(nn.Module):                      # TS: Wang et al. (2017)
    def __init__(self, c_in=9, K=6):
        super().__init__()
        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding="same"), nn.BatchNorm1d(o), nn.ReLU())
        self.f = nn.Sequential(blk(c_in, 128, 8), blk(128, 256, 5), blk(256, 128, 3))
        self.out = nn.Linear(128, K)

    def forward(self, x):
        return self.out(self.f(x).mean(-1))


class GCN(nn.Module):                      # GRA: 2 layers, hidden 64, ReLU, dropout 0.5 (Kipf & Welling)
    def __init__(self, d_in=1433, K=7, h=64, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(d_in, h), nn.Linear(h, K), p

    def forward(self, x, A):               # A: normalised sparse adjacency with self-loops
        x = F.dropout(x, self.p, self.training)
        x = F.relu(torch.sparse.mm(A, self.l1(x)))
        x = F.dropout(x, self.p, self.training)
        return torch.sparse.mm(A, self.l2(x))


def gcn_adjacency(edge_index, n, device):
    """D^{-1/2} (A + I) D^{-1/2} as a sparse tensor; edges are made symmetric and deduplicated."""
    ei = torch.as_tensor(edge_index, dtype=torch.long)
    ei = torch.cat([ei, ei.flip(0), torch.arange(n).repeat(2, 1)], 1)
    ei = torch.unique(ei, dim=1)
    w = torch.ones(ei.shape[1])
    deg = torch.zeros(n).index_add_(0, ei[0], w)
    norm = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()
    return torch.sparse_coo_tensor(ei, norm, (n, n)).coalesce().to(device)


class AudioCNN(nn.Module):                 # AUD: 4 x [Conv3x3-BN-ReLU-MaxPool2], C = 32..256, GAP, dropout 0.3
    def __init__(self, K=10):
        super().__init__()
        layers, c = [], 1
        for o in (32, 64, 128, 256):
            layers += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
            c = o
        self.f = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(256, K)

    def forward(self, x):                  # x: (B, 64, frames)
        x = self.f(x.unsqueeze(1)).mean((2, 3))
        return self.out(self.drop(x))


class LinearHead(nn.Module):               # H3: multinomial logistic regression on frozen embeddings
    def __init__(self, d, K):
        super().__init__()
        self.lin = nn.Linear(d, K)

    def forward(self, x):
        return self.lin(x)


# ----------------------------------------------------------------------------- log-mel features
def _hz_to_mel(f):
    return 2595.0 * torch.log10(torch.tensor(1.0) + f / 700.0)


def mel_filterbank(sr=16000, n_fft=400, n_mels=64, f_min=0.0, f_max=None):
    """HTK mel scale, no area normalisation (same convention as torchaudio's defaults)."""
    f_max = f_max or sr / 2
    n_freq = n_fft // 2 + 1
    all_f = torch.linspace(0, sr / 2, n_freq)
    m_pts = torch.linspace(_hz_to_mel(torch.tensor(f_min)), _hz_to_mel(torch.tensor(f_max)), n_mels + 2)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_f.unsqueeze(1)
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return torch.clamp(torch.min(down, up), min=0.0)          # (n_freq, n_mels)


def log_mel(wave_batch, fb, n_fft=400, hop=160):
    """wave_batch: (B, 64000) at 16 kHz -> (B, 64, 401): Hann window, centred reflect padding, power 2."""
    win = torch.hann_window(n_fft, device=wave_batch.device)
    S = torch.stft(wave_batch, n_fft, hop_length=hop, win_length=n_fft, window=win, center=True,
                   pad_mode="reflect", return_complex=True).abs() ** 2       # (B, n_freq, frames)
    mel = torch.matmul(S.transpose(1, 2), fb.to(wave_batch.device)).transpose(1, 2)
    return torch.log(mel + 1e-6)
