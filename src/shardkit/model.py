"""The reward model, and the tiny stand-in that lets every test run on a laptop.

Both come out of one builder so the code path under test is the code path that trains. The
scalar head is the reason this benchmark is about memory rather than about logits: a causal
LM at this batch would hold a vocab-sized activation of several gigabytes, which would
dominate the memory story and hide the thing being measured.
"""

from __future__ import annotations

import torch
from torch import nn

from shardkit.config import ModelConfig


class TinyRewardModel(nn.Module):
    """A few real transformer blocks with a scalar head. Small enough for a CPU test."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.tiny_vocab, cfg.tiny_hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.tiny_hidden,
            nhead=4,
            dim_feedforward=cfg.tiny_hidden * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.tiny_hidden,
                    nhead=4,
                    dim_feedforward=cfg.tiny_hidden * 4,
                    batch_first=True,
                    dropout=0.0,
                )
                for _ in range(cfg.tiny_layers)
            ]
        )
        del layer
        self.norm = nn.LayerNorm(cfg.tiny_hidden)
        self.score = nn.Linear(cfg.tiny_hidden, 1)

    @property
    def layers(self) -> list[nn.Module]:
        """The units FSDP shards. Naming them here keeps parallel.py model-agnostic."""
        return list(self.blocks)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        pad = attention_mask == 0
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=pad)
        hidden = self.norm(hidden)
        # Score the last real token, the same convention a sequence-classification head uses.
        last = attention_mask.sum(dim=1).clamp(min=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        scores: torch.Tensor = self.score(pooled).squeeze(-1)
        return scores


def build(cfg: ModelConfig) -> nn.Module:
    """Return the tiny model, or the real one behind an optional import."""
    if cfg.tiny:
        return TinyRewardModel(cfg)

    from transformers import AutoConfig, AutoModelForSequenceClassification

    hf_config = AutoConfig.from_pretrained(cfg.name, num_labels=1)
    hf_config.pad_token_id = getattr(hf_config, "pad_token_id", None) or 0
    model: nn.Module = AutoModelForSequenceClassification.from_pretrained(
        cfg.name, config=hf_config, dtype=torch.float32
    )
    return model


def transformer_layers(model: nn.Module) -> list[nn.Module]:
    """The repeated blocks, whatever the model calls them.

    FSDP wants one unit per block. Getting this wrong is silent: wrap nothing and the whole
    model is a single unit, so the all-gather is the entire parameter set at once.
    """
    if hasattr(model, "layers"):
        layers = model.layers
        if isinstance(layers, (list, nn.ModuleList)):
            return list(layers)
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        node: object = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, (list, nn.ModuleList)):
            return list(node)
    raise ValueError(f"cannot find transformer blocks on {type(model).__name__}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
