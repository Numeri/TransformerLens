"""Hooked Transformer Bert Pooled Text Embedding Head Component.

This module contains all the component :class:`BertPooledTextEmbedding`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class BertPooledTextEmbedding(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.dense = nn.Linear(self.cfg.d_model, self.cfg.d_model)
        self.activation = nn.Tanh()

    def forward(self, resid: Float[torch.Tensor, "batch pos d_model"]) -> torch.Tensor:
        first_token_tensor = resid[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
