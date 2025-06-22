# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class BlockPress(BasePress):
    """
    Simulates block prompt processing.
    Segments input sequence into non-overlapping blocks and compresses iteratively.
    Keeps limited memory overhead for long context inference.
    """

    press: ScorerPress
    block_size: int = 128

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "BlockPress requires a ScorerPress"

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values

        assert attentions is None, "BlockPress does not support attentions."

        bsz, num_key_value_heads, q_len, head_dim = keys.shape

        block_size = self.block_size if self.block_size < q_len else q_len
        n_kept = int(q_len * (1 - self.compression_ratio))

        kept_indices = torch.empty((bsz, num_key_value_heads, 0), dtype=torch.long, device=keys.device)
        states = hidden_states.view(bsz, q_len, num_key_value_heads, -1).transpose(1, 2)

        for i in range(0, q_len, block_size):
            current_indices = torch.arange(i, min(i+block_size, q_len), device=keys.device)
            current_indices = current_indices.expand(bsz, num_key_value_heads, -1)
            current_indices = torch.cat([kept_indices, current_indices], dim=2)

            if current_indices.shape[-1] <= n_kept:
                # Not evict when cache budget is not met
                kept_indices = current_indices
            else:
                current_indices = current_indices.unsqueeze(-1)
                current_states = states.gather(2, current_indices.expand(-1, -1, -1, states.shape[-1]))
                current_states = current_states.transpose(1, 2).reshape(bsz, -1, hidden_states.shape[-1])

                scores = self.press.score(
                    module,
                    current_states,
                    keys.gather(2, current_indices.expand(-1, -1, -1, head_dim)),
                    values.gather(2, current_indices.expand(-1, -1, -1, head_dim)),
                    attentions,
                    kwargs,
                )
                topk_indices = scores.topk(n_kept, dim=-1).indices
                kept_indices = current_indices.squeeze(-1).gather(2, topk_indices)

        final_indices = kept_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        keys = keys.gather(2, final_indices).contiguous()
        values = values.gather(2, final_indices).contiguous()

        return keys, values


@dataclass
class KeyDiffPress(ScorerPress):
    """
    KeyDiff (https://arxiv.org/abs/2504.15364) evict tokens based solely on key similarity.
    """
    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        anchor = F.normalize(keys, p=2, dim=-1).mean(dim=2, keepdim=True)

        return -F.cosine_similarity(keys, anchor, dim=-1)
