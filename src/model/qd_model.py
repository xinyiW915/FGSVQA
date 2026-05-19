# ----------------------------
# Model: CLIP encoder + w_art/w_str pooling -> weighted sequences -> seq model -> head fusion
# ----------------------------
import torch
import torch.nn as nn

from .clip_dense_encoder import CLIPDenseEncoder


def weighted_pool_2d(fmap, wmap, eps=1e-6):
    """
    fmap: [B, C, H, W]
    wmap: [B, 1, H, W]
    -> [B, C]
    """
    w = wmap.clamp(0.0, 1.0)
    w = w / (w.sum(dim=(2, 3), keepdim=True) + eps)
    return (fmap * w).sum(dim=(2, 3))

def default_gate_stats(w_art, w_str, fmap_bt, eps=1e-6):
    """
    w_art:   [B, 1, T, H, W]
    w_str:   [B, 1, T, H, W]
    fmap_bt: [B, T, C, H, W]
    -> [B, 3]
    """
    mu_art = w_art.mean(dim=(1, 2, 3, 4))
    mu_str = w_str.mean(dim=(1, 2, 3, 4))
    mu_raw = fmap_bt.abs().mean(dim=(1, 2, 3, 4))

    stats = torch.stack([mu_art, mu_str, mu_raw], dim=1)
    return stats

def two_gate_stats(w_art, w_str):
    # [B, 2]
    mu_art = w_art.mean(dim=(1, 2, 3, 4))
    mu_str = w_str.mean(dim=(1, 2, 3, 4))
    return torch.stack([mu_art, mu_str], dim=1)

class QD_MODEL(nn.Module):
    def __init__(
        self,
        *,
        clip_model="openai/clip-vit-base-patch16",
        head_hidden=384,
        gate_hidden=32,
        head_dropout=0.2,
        gate_dropout=0.1,
        ablation_mode="full",  # "full" | "art" | "str" | "raw"
    ):
        super().__init__()
        self.ablation_mode = ablation_mode

        self.encoder = CLIPDenseEncoder(model_name=str(clip_model))
        c = int(self.encoder.hidden_size)

        # 3 heads
        self.head_art = nn.Sequential(
            nn.Linear(c, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 2, 1),
        )
        self.head_str = nn.Sequential(
            nn.Linear(c, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 2, 1),
        )
        self.head_fmap = nn.Sequential(
            nn.Linear(c, head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 2, 1),
        )
        # full model: 3-way gate
        self.gate = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden, 3),
        )
        # ablation: two gates only
        self.gate_two = nn.Sequential(
            nn.Linear(2, gate_hidden),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden, 2),
        )

    def freeze_clip_all(self):
        self.encoder.freeze_all()

    def unfreeze_clip_last_blocks(self, n_blocks=2, also_unfreeze_ln=True):
        self.encoder.unfreeze_last_blocks(
            n_blocks=n_blocks,
            also_unfreeze_ln=also_unfreeze_ln,
        )

    def temporal_pool(self, x):
        """
        x: [B, T, C]
        return: [B, C]
        """
        return x.mean(dim=1)

    def forward(self, rgb, w_art, w_str, gate_stats=None):
        """
        rgb: [B,3,T,H,W]   (rgb in [0,1])
        w_art/w_str: [B,1,T,H,W]   (0..1)
        gate_stats: [B,3]
        """
        B, _C, T, H, W = rgb.shape

        x2d = rgb.permute(0, 2, 1, 3, 4).contiguous().view(B * T, 3, H, W)
        fmap2d = self.encoder(x2d)
        _, C, Hp, Wp = fmap2d.shape  # (B*T, 768, 14, 14)

        w_art_bt = w_art.transpose(1, 2)  # (B, T, 1, 14, 14)
        w_str_bt = w_str.transpose(1, 2)
        fmap_bt = fmap2d.view(B, T, C, Hp, Wp)  # (B, T, 768, 14, 14)

        z_art = torch.stack([weighted_pool_2d(fmap_bt[:, i], w_art_bt[:, i]) for i in range(T)], dim=1)  # (B, T, 768)
        z_str = torch.stack([weighted_pool_2d(fmap_bt[:, i], w_str_bt[:, i]) for i in range(T)], dim=1)
        z_raw = fmap_bt.mean(dim=(-2, -1))

        h_art = self.temporal_pool(z_art)  # [B, C]
        h_str = self.temporal_pool(z_str)
        h_fmap = self.temporal_pool(z_raw)

        q_art = self.head_art(h_art)
        q_str = self.head_str(h_str)
        q_fmap = self.head_fmap(h_fmap)

        mode = self.ablation_mode.lower()
        if mode == "art":
            y_hat = q_art
            weights = torch.tensor([1.0, 0.0, 0.0], device=q_art.device).unsqueeze(0).repeat(B, 1)
        elif mode == "str":
            y_hat = q_str
            weights = torch.tensor([0.0, 1.0, 0.0], device=q_art.device).unsqueeze(0).repeat(B, 1)
        elif mode == "raw":
            y_hat = q_fmap
            weights = torch.tensor([0.0, 0.0, 1.0], device=q_art.device).unsqueeze(0).repeat(B, 1)
        elif mode == "art+str":
            if gate_stats is None:
                gate_stats = two_gate_stats(w_art, w_str)  # [B, 2]
            g_ar = self.gate_two(gate_stats)  # [B, 2]
            a, b_ = torch.softmax(g_ar, dim=1).split(1, dim=1)
            y_hat = a * q_art + b_ * q_str
            zero = torch.zeros_like(a)
            weights = torch.cat([a, b_, zero], dim=1)
        elif mode == "full":
            if gate_stats is None:
                gate_stats = default_gate_stats(w_art, w_str, fmap_bt)
            g = self.gate(gate_stats)
            a, b_, c_ = torch.softmax(g, dim=1).split(1, dim=1)
            y_hat = a * q_art + b_ * q_str + c_ * q_fmap
            weights = torch.cat([a, b_, c_], dim=1)
        else:
            raise ValueError(f"Unknown ablation_mode: {self.ablation_mode}")

        aux = (
            q_art.squeeze(1),
            q_str.squeeze(1),
            q_fmap.squeeze(1),
            weights,
        )
        return y_hat.squeeze(1), aux