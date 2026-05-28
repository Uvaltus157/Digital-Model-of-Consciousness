from __future__ import annotations

"""
symbolic_report_language.py

Internal symbolic reporting layer for the M5 memory-thought stack.

Goal:
    latent internal states
        -> internal symbolic code
        -> phoneme / sound-code sequence
        -> human-readable report tokens

This is NOT a real pretrained language model.
It is a lightweight trainable interface that gives the agent a structured
"inner report" channel.

Main classes:
    InternalSymbolicCodec
    PhonemeCodeHead
    HumanLanguageReportHead
    InnerSpeechLoop
    SymbolicReportMixin

The intended integration point:
    workspace, thought_sequence, reflection, object_repr, memory_context,
    attention weights, action logits
        -> InnerSpeechLoop
        -> symbolic_report dict
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SymbolicReportConfig:
    input_dim: int = 1024
    symbol_dim: int = 256
    symbol_vocab_size: int = 512
    phoneme_vocab_size: int = 96
    text_vocab_size: int = 2048
    max_symbols: int = 12
    max_phonemes: int = 48
    max_text_tokens: int = 48
    inner_speech_steps: int = 3


class InternalSymbolicCodec(nn.Module):
    """
    Converts a continuous latent state into internal discrete-like symbols.

    Output:
        symbol_logits: [B, max_symbols, symbol_vocab]
        symbol_probs:  [B, max_symbols, symbol_vocab]
        symbol_embed:  [B, max_symbols, symbol_dim]

    It uses soft symbols so it remains differentiable.
    """
    def __init__(
        self,
        input_dim: int,
        symbol_dim: int = 256,
        symbol_vocab_size: int = 512,
        max_symbols: int = 12,
    ) -> None:
        super().__init__()
        self.max_symbols = max_symbols
        self.symbol_vocab_size = symbol_vocab_size
        self.symbol_dim = symbol_dim

        self.to_slots = nn.Sequential(
            nn.Linear(input_dim, symbol_dim * max_symbols),
            nn.ReLU(inplace=True),
            nn.LayerNorm(symbol_dim * max_symbols),
        )
        self.symbol_logits = nn.Linear(symbol_dim, symbol_vocab_size)
        self.symbol_embedding = nn.Embedding(symbol_vocab_size, symbol_dim)

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = latent.shape[0]
        slots = self.to_slots(latent).view(b, self.max_symbols, self.symbol_dim)
        logits = self.symbol_logits(slots)
        probs = torch.softmax(logits, dim=-1)

        # Soft embedding: differentiable internal code
        embed_weight = self.symbol_embedding.weight
        symbol_embed = torch.matmul(probs, embed_weight)

        hard_ids = torch.argmax(logits, dim=-1)

        return {
            "symbol_logits": logits,
            "symbol_probs": probs,
            "symbol_embed": symbol_embed,
            "symbol_ids": hard_ids,
        }


class PhonemeCodeHead(nn.Module):
    """
    Translates internal symbols into phoneme/sound-code logits.

    This is not TTS. It produces symbolic phoneme-like codes that can later
    be mapped to real speech or text.
    """
    def __init__(
        self,
        symbol_dim: int = 256,
        phoneme_vocab_size: int = 96,
        max_phonemes: int = 48,
    ) -> None:
        super().__init__()
        self.max_phonemes = max_phonemes
        self.query = nn.Parameter(torch.randn(1, max_phonemes, symbol_dim) * 0.02)
        self.attn = nn.MultiheadAttention(symbol_dim, num_heads=4, batch_first=True)
        self.head = nn.Linear(symbol_dim, phoneme_vocab_size)

    def forward(self, symbol_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = symbol_embed.shape[0]
        q = self.query.expand(b, -1, -1)
        decoded, attn = self.attn(q, symbol_embed, symbol_embed, need_weights=True)
        logits = self.head(decoded)
        ids = torch.argmax(logits, dim=-1)
        return {
            "phoneme_logits": logits,
            "phoneme_ids": ids,
            "phoneme_attention": attn,
        }


class HumanLanguageReportHead(nn.Module):
    """
    Produces text-token logits from internal symbol embeddings and phoneme codes.

    This is a lightweight trainable report generator.
    It does not know Russian/English by itself unless trained or mapped with a
    vocabulary outside the model.
    """
    def __init__(
        self,
        symbol_dim: int = 256,
        phoneme_vocab_size: int = 96,
        text_vocab_size: int = 2048,
        max_text_tokens: int = 48,
    ) -> None:
        super().__init__()
        self.max_text_tokens = max_text_tokens
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, symbol_dim)
        self.query = nn.Parameter(torch.randn(1, max_text_tokens, symbol_dim) * 0.02)
        self.attn_symbols = nn.MultiheadAttention(symbol_dim, num_heads=4, batch_first=True)
        self.attn_phonemes = nn.MultiheadAttention(symbol_dim, num_heads=4, batch_first=True)
        self.mix = nn.Sequential(
            nn.Linear(symbol_dim * 2, symbol_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(symbol_dim),
        )
        self.text_head = nn.Linear(symbol_dim, text_vocab_size)

    def forward(self, symbol_embed: torch.Tensor, phoneme_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = symbol_embed.shape[0]
        q = self.query.expand(b, -1, -1)
        ph = self.phoneme_embedding(phoneme_ids)

        sym_ctx, sym_attn = self.attn_symbols(q, symbol_embed, symbol_embed, need_weights=True)
        ph_ctx, ph_attn = self.attn_phonemes(q, ph, ph, need_weights=True)

        x = self.mix(torch.cat([sym_ctx, ph_ctx], dim=-1))
        logits = self.text_head(x)
        ids = torch.argmax(logits, dim=-1)

        return {
            "text_logits": logits,
            "text_token_ids": ids,
            "symbol_text_attention": sym_attn,
            "phoneme_text_attention": ph_attn,
        }


class InnerSpeechLoop(nn.Module):
    """
    Full inner report pathway.

    Inputs are latent states of the conscious model.
    It first compresses them into a report latent, then creates:
        internal symbols
        phoneme codes
        text token codes
    """
    def __init__(self, cfg: SymbolicReportConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.latent_mixer = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.symbol_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.symbol_dim),
        )
        self.recurrent_thought = nn.GRUCell(cfg.symbol_dim, cfg.symbol_dim)

        self.codec = InternalSymbolicCodec(
            input_dim=cfg.symbol_dim,
            symbol_dim=cfg.symbol_dim,
            symbol_vocab_size=cfg.symbol_vocab_size,
            max_symbols=cfg.max_symbols,
        )
        self.phonemes = PhonemeCodeHead(
            symbol_dim=cfg.symbol_dim,
            phoneme_vocab_size=cfg.phoneme_vocab_size,
            max_phonemes=cfg.max_phonemes,
        )
        self.language = HumanLanguageReportHead(
            symbol_dim=cfg.symbol_dim,
            phoneme_vocab_size=cfg.phoneme_vocab_size,
            text_vocab_size=cfg.text_vocab_size,
            max_text_tokens=cfg.max_text_tokens,
        )

        self.confidence = nn.Sequential(nn.Linear(cfg.symbol_dim, 1), nn.Sigmoid())

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.latent_mixer(latent)
        x = h
        inner_states = []
        for _ in range(self.cfg.inner_speech_steps):
            h = self.recurrent_thought(x, h)
            inner_states.append(h)

        report_latent = h
        inner_sequence = torch.stack(inner_states, dim=1)

        symbolic = self.codec(report_latent)
        phoneme = self.phonemes(symbolic["symbol_embed"])
        language = self.language(symbolic["symbol_embed"], phoneme["phoneme_ids"])

        return {
            "report_latent": report_latent,
            "inner_speech_sequence": inner_sequence,
            "confidence": self.confidence(report_latent),
            **symbolic,
            **phoneme,
            **language,
        }


class SymbolicReportMixin:
    """
    Mixin for a ConsciousDreamer-like model.

    Expected tensors from the base model output:
        workspace_out
        thoughts.thought
        reflection_out.reflection
        object_repr
        memory.memory_context
        values.value_latent
        embodied_targets
        hand_ctrl

    Usage:
        self.init_symbolic_report(...)
        report = self.symbolic_report_from_out(out)
    """
    def init_symbolic_report(
        self,
        workspace_dim: int,
        thought_dim: int,
        reflective_dim: int,
        object_dim: int,
        memory_dim: int,
        value_dim: int,
        embodied_dim: int,
        hand_motor_dim: int,
        symbol_dim: int = 256,
        symbol_vocab_size: int = 512,
        phoneme_vocab_size: int = 96,
        text_vocab_size: int = 2048,
    ) -> None:
        input_dim = (
            workspace_dim
            + thought_dim
            + reflective_dim
            + object_dim
            + memory_dim
            + value_dim
            + embodied_dim
            + hand_motor_dim
        )
        self.symbolic_report_cfg = SymbolicReportConfig(
            input_dim=input_dim,
            symbol_dim=symbol_dim,
            symbol_vocab_size=symbol_vocab_size,
            phoneme_vocab_size=phoneme_vocab_size,
            text_vocab_size=text_vocab_size,
        )
        self.inner_speech = InnerSpeechLoop(self.symbolic_report_cfg)

    def symbolic_report_from_out(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        memory_context = out.get("memory", {}).get("memory_context")
        if memory_context is None:
            memory_context = torch.zeros_like(out["workspace_out"])

        latent = torch.cat(
            [
                out["workspace_out"],
                out["thoughts"]["thought"],
                out["reflection_out"]["reflection"],
                out["object_repr"],
                memory_context,
                out["values"]["value_latent"],
                out["embodied_targets"],
                out["hand_ctrl"],
            ],
            dim=-1,
        )
        return self.inner_speech(latent)


# Optional tiny vocabulary helper for debugging.
# This does not make a real language model; it only makes reports readable.
DEFAULT_DEBUG_TOKENS_RU = [
    "<pad>", "<bos>", "<eos>",
    "вижу", "объект", "рука", "касание", "сила", "движение",
    "неуверенность", "память", "план", "приблизиться", "исследовать",
    "держать", "отпустить", "слева", "справа", "форма", "тяжесть",
    "новизна", "цель", "я", "мир", "думаю", "проверяю",
]


def decode_debug_tokens(token_ids: torch.Tensor, vocab: Optional[list[str]] = None, max_tokens: int = 32) -> list[str]:
    """
    Converts token ids to a rough debug string using a tiny provided vocabulary.

    This is only for visual debugging before training a real tokenizer/decoder.
    """
    if vocab is None:
        vocab = DEFAULT_DEBUG_TOKENS_RU

    ids = token_ids.detach().cpu()
    lines = []
    for row in ids:
        words = []
        for i in row[:max_tokens].tolist():
            words.append(vocab[int(i) % len(vocab)])
        lines.append(" ".join(words))
    return lines
