from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ObjectImageryConfig:
    object_dim: int = 128
    workspace_dim: int = 256
    thought_dim: int = 192
    reflection_dim: int = 192
    hidden_dim: int = 256
    image_size: int = 96
    image_channels: int = 3
    use_context: bool = True
    shape_classes: int = 8
    color_classes: int = 12
    material_classes: int = 8


class ObjectImageryDecoder(nn.Module):
    def __init__(self, cfg: Optional[ObjectImageryConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or ObjectImageryConfig()

        in_dim = self.cfg.object_dim
        if self.cfg.use_context:
            in_dim += self.cfg.workspace_dim + self.cfg.thought_dim + self.cfg.reflection_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, self.cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.cfg.hidden_dim),
        )

        self.side = self.cfg.image_size // 8
        self.to_grid = nn.Linear(self.cfg.hidden_dim, 128 * self.side * self.side)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 96, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.rgb_head = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.cfg.image_channels, 1),
            nn.Sigmoid(),
        )
        self.alpha_head = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self.shape_head = nn.Linear(self.cfg.hidden_dim, self.cfg.shape_classes)
        self.color_head = nn.Linear(self.cfg.hidden_dim, self.cfg.color_classes)
        self.material_head = nn.Linear(self.cfg.hidden_dim, self.cfg.material_classes)
        self.object_confidence = nn.Sequential(nn.Linear(self.cfg.hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        object_repr: torch.Tensor,
        workspace: Optional[torch.Tensor] = None,
        thought: Optional[torch.Tensor] = None,
        reflection: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        x_parts = [object_repr]
        if self.cfg.use_context:
            b = object_repr.shape[0]
            device = object_repr.device
            dtype = object_repr.dtype

            def z(t: Optional[torch.Tensor], dim: int):
                if t is None:
                    return torch.zeros(b, dim, device=device, dtype=dtype)
                return t

            x_parts.extend([
                z(workspace, self.cfg.workspace_dim),
                z(thought, self.cfg.thought_dim),
                z(reflection, self.cfg.reflection_dim),
            ])

        x = torch.cat(x_parts, dim=-1)
        latent = self.encoder(x)
        grid = self.to_grid(latent).view(latent.shape[0], 128, self.side, self.side)
        feat = self.up(grid)

        shape_logits = self.shape_head(latent)
        color_logits = self.color_head(latent)
        material_logits = self.material_head(latent)

        return {
            "latent": latent,
            "rgb": self.rgb_head(feat),
            "alpha": self.alpha_head(feat),
            "depth": self.depth_head(feat),
            "shape_logits": shape_logits,
            "color_logits": color_logits,
            "material_logits": material_logits,
            "shape_id": torch.argmax(shape_logits, dim=-1),
            "color_id": torch.argmax(color_logits, dim=-1),
            "material_id": torch.argmax(material_logits, dim=-1),
            "object_confidence": self.object_confidence(latent),
        }


def blended_object_image(rgb: torch.Tensor, alpha: torch.Tensor, background: float = 0.08) -> torch.Tensor:
    bg = torch.full_like(rgb, float(background))
    return rgb * alpha + bg * (1.0 - alpha)
