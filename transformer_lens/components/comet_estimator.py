"""Hooked Transformer CometKiwi score estimator head.

This module contains the component :class:`CoemtEstimator`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.hook_points import HookPoint


class CometEstimator(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)

        # Layerwise attention over output of the embedding layer and all encoder layers
        self.layerwise_attention = LayerwiseAttention(cfg.n_layers + 1)
        self.dense_up = nn.Linear(self.cfg.d_model, self.cfg.comet_estimator_width, bias=True)
        self.activation1 = nn.Tanh()
        self.dense_down = nn.Linear(self.cfg.comet_estimator_width, self.cfg.d_model, bias=True)
        self.activation2 = nn.Tanh()
        self.dense_estimator = nn.Linear(self.cfg.d_model, 1, bias=True)

        self.hook_input = HookPoint()  # [batch, num_layers, d_model]
        self.hook_layerwise_attention = HookPoint()  # [batch, d_model]
        self.hook_dense_up = HookPoint()  # [batch, comet_estimator_width]
        self.hook_dense_up_act = HookPoint()  # [batch, comet_estimator_width]
        self.hook_dense_down = HookPoint()  # [batch, d_model]
        self.hook_dense_down_act = HookPoint()  # [batch, d_model]
        self.hook_comet_score = HookPoint()  # [batch, 1]

    def forward(self, first_token_activations: Float[torch.Tensor, "batch num_layers d_model"]) -> torch.Tensor:
        x = self.hook_input(first_token_activations)
        x = self.hook_layerwise_attention(self.layerwise_attention(x))

        x = self.hook_dense_up(self.dense_up(x))
        x = self.hook_dense_up_act(self.activation1(x))
        x = self.hook_dense_down(self.dense_down(x))
        x = self.hook_dense_down_act(self.activation2(x))

        x = self.hook_comet_score(self.dense_estimator(x).squeeze(0))

        return x


class LayerwiseAttention(nn.Module):
    def __init__(self, num_layers: int, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.gamma = nn.Parameter(torch.tensor([1.0], **factory_kwargs))
        self.layer_weights = nn.Parameter(torch.empty((num_layers,), **factory_kwargs))

    def forward(self, first_token_activations: Float[torch.Tensor, "batch num_layers d_model"]) -> Float[torch.Tensor, "batch d_model"]:
        x = self.layer_weights.unsqueeze(-1) * first_token_activations  # [batch, num_layers, d_model]
        x = self.gamma * x.sum(dim=1)  # [batch, d_model]

        return x
