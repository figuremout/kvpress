
import logging
from dataclasses import dataclass

import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.phi3.modeling_phi3 import Phi3Attention

from kvpress.presses.base_press import BasePress
from kvpress.presses.think_press import ThinKPress
from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress


logger = logging.getLogger(__name__)


@dataclass
class ScorerSizePress(BasePress):
    """
    ScorerPress based on fixed KV Cache size rather than compression ratio.
    """
    cache_size: int = 1024

    def __post_init__(self):
        assert self.cache_size > 0, "Cache size must be greater than 0"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute a tensor of scores with shape (bsz, num_key_value_heads, q_len)
        The KV pairs with lowest scores will be pruned in the `compress` method.
        """
        raise NotImplementedError

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        n_kept = min(self.cache_size, scores.size(-1))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values

    @property
    def compression_ratio(self):
        return 0.0

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")


@dataclass
class Compose_ThinKPress(BasePress):
    """
    Chain multiple presses together to create a composed press
    """

    press: BasePress
    key_channel_compression_ratio: float = 0.0

    def __post_init__(self):
        assert not isinstance(self.press, (ObservedAttentionPress, AdaKVPress)), "ComposedPress cannot contains ObservedAttentionPress or AdaKVPress"
        assert 0 <= self.key_channel_compression_ratio < 1, "Key channel compression ratio must be between 0 and 1"

    def forward_hook(self, module, input, kwargs, output):
        presses = [self.press, ThinKPress(key_channel_compression_ratio=self.key_channel_compression_ratio)]
        for press in presses:
            output = press.forward_hook(module, input, kwargs, output)
        return output

    @property
    def compression_ratio(self):
        return 0.0

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")


@dataclass
class KCMSizePress(ScorerSizePress): # 驱逐 key channel matters 的 token
    cache_size: int = 1024
    window_size: int = 64 # 和 SnapKV 保持一致
    channel_window_size: int = 16
    kernel_size: int = 5 # TODO remove

    def compute_window_queries(self, module, hidden_states, position_embeddings): # Copy from ThinKPress
        """
        Re-compute the last window_size query states
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim

        # Get last self.window_size queries
        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states[:, -self.window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        elif hasattr(module, "q_proj"):
            # Assume Llama-like attention layer
            query_states = module.q_proj(hidden_states[:, -self.window_size:])
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, self.window_size, num_heads, head_dim).transpose(1, 2)

        # Support for Qwen3 and Gemma3 QK norm
        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            query_states = module.q_norm(query_states)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -self.window_size :], sin[:, -self.window_size :]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        return query_states

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        # Compute scores per dimension
        bsz, num_key_value_heads, q_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        # (B, nh * groups, window, hs) Q 自己算的
        queries = self.compute_window_queries(module, kwargs["hidden_states"], kwargs["position_embeddings"])
        queries_norm = torch.pow(queries, 2)
        # (B, nh, hs)
        queries_norm = queries_norm.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, module.head_dim).mean(2) # (B, nh, window, hs)
        # (B, nh, T, hs)
        keys_norm = torch.pow(keys, 2)

        if self.channel_window_size > 0:
            key_scores = keys_norm[..., :self.channel_window_size] @ queries_norm[..., :self.channel_window_size].transpose(-1, -2) # (B, nh, T, window)
        else:
            key_scores = keys_norm @ queries_norm.transpose(-1, -2) # (B, nh, T, window)
        # key_scores = key_scores[..., :-self.window_size, :] # (B, nh, T-window, window)
        scores = key_scores.mean(-1) # # (B, nh, T-window)

        # if self.is_pool:
        #     scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        # Add back the observation window. Use max score to make sure the window is not pruned.
        # scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return -scores

@dataclass
class SnapKVSizePress(ScorerSizePress):
    """
    SnapKV (https://arxiv.org/abs/2404.14469) use the attention of the latest window_size tokens to estimate the
    importance of the previous KV pairs. We use the default settings from:
    https://github.com/FasterDecoding/SnapKV/blob/main/snapkv/monkeypatch/snapkv_utils.py#L24
    """

    cache_size: int = 1024
    window_size: int = 64
    kernel_size: int = 5

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        if isinstance(module, Phi3Attention):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        elif hasattr(module, "q_proj"):
            # Assume Llama-like attention layer
            query_states = module.q_proj(hidden_states[:, -window_size:])
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # Support for Qwen3 and Gemma3 QK norm
        if isinstance(module, (Qwen3Attention, Gemma3Attention)):
            query_states = module.q_norm(query_states)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.mean(2)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores


@dataclass
class H2OSizePress(ScorerSizePress):
    """
    原 kvpress 实现 ObservedAttentionPress 占用太多显存，手动计算
    """

    cache_size: int = 1024
    block_size: int = 2048 # Smaller block size trades increased execution time for reduced memory overhead.
                           # Different block sizes may slightly affect scores due to numerical errors.

    @staticmethod
    def compute_block_attention(module, hidden_states, keys, position_embeddings, start, end):
        """
        逐块计算注意力权重并立即 sum，避免存储整个 attention matrix
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, start:end])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, start:end])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        # Q shape: (B, nh, block_size, hs)
        block_size = end - start
        query_states = query_states.view(bsz, block_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, start:end], sin[:, start:end]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim) # (B, nh, block_size, T)
        
        block_mask = torch.triu(
            torch.full((block_size, q_len), float("-inf"), device=attn_weights.device),
            diagonal=start+1)
        attn_weights += block_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # (B, nh, block_size, T)
        attn_weights = attn_weights.view(bsz, num_key_value_groups, module.config.num_key_value_heads, block_size, q_len).mean(1)
        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        bsz, num_key_value_heads, n_tokens, _ = keys.shape
        
        scores = torch.zeros(bsz, num_key_value_heads, n_tokens, device=keys.device)

        for start in range(0, n_tokens, self.block_size):
            end = min(start + self.block_size, n_tokens)
            block_attentions = self.compute_block_attention(
                module, hidden_states, keys, kwargs["position_embeddings"],
                start, end)
            scores += block_attentions.sum(2)
        n_tokens_in_sum = torch.arange(n_tokens, 0, -1).to(scores.device, scores.dtype)
        scores = scores / n_tokens_in_sum
        return scores


@dataclass
class KeyDiffSizePress(ScorerSizePress):
    """
    KeyDiff (https://arxiv.org/abs/2504.15364) evict tokens based solely on key similarity.
    """
    cache_size: int = 1024
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