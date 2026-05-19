import torch
import torch.nn as nn
from transformers import CLIPModel
from transformers.utils import logging
logging.set_verbosity_error()


class CLIPDenseEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(str(model_name), use_safetensors=True)

        self.vision = self.clip.vision_model
        vcfg = self.clip.config.vision_config
        self.hidden_size = int(vcfg.hidden_size)
        self.patch_size = int(vcfg.patch_size)
        self.image_size = int(vcfg.image_size)

        # OpenAI CLIP normalization constants
        self.register_buffer("_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x01):
        x = (x01 - self._mean.to(dtype=x01.dtype)) / self._std.to(dtype=x01.dtype)
        out = self.vision(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:, :]  # [B, 1+N, C] drop CLS -> [B, N, C]
        b, n, c = tokens.shape
        side = int(n**0.5)
        if side * side != n:
            raise RuntimeError(f"CLIP patch tokens N={n} not a square; cannot reshape to 2D grid.")

        fmap = tokens.transpose(1, 2).contiguous().view(b, c, side, side)
        return fmap

    def freeze_all(self):
        for p in self.clip.parameters():
            p.requires_grad = False

    def unfreeze_last_blocks(self, n_blocks=2, also_unfreeze_ln=True):
        self.freeze_all()
        layers = self.vision.encoder.layers
        n_total = len(layers)

        # all unfreeze
        if n_blocks < 0 or n_blocks >= n_total:
            for p in self.vision.parameters():
                p.requires_grad = True
        # unfreeze n blocks
        else:
            k = max(0, min(int(n_blocks), n_total))
            for i in range(n_total - k, n_total):
                for p in layers[i].parameters():
                    p.requires_grad = True

        if also_unfreeze_ln:
            if hasattr(self.vision, "pre_layrnorm"):
                for p in self.vision.pre_layrnorm.parameters():
                    p.requires_grad = True
            if hasattr(self.vision, "post_layernorm"):
                for p in self.vision.post_layernorm.parameters():
                    p.requires_grad = True
