import torch
from torch import nn


class _EmbeddingModel(nn.Module):
    """Change only the local embedding and its projection into fixed-width context."""

    def __init__(
        self, width=32, activation="linear", adapter="identity", normalize=False
    ):
        super().__init__()
        self.sensor = nn.Linear(4, width)
        self.activation = nn.GELU() if activation == "gelu" else nn.Identity()
        self.local_norm = nn.LayerNorm(width) if normalize else nn.Identity()
        if adapter == "identity":
            if width != 32:
                raise ValueError("identity adapter requires width 32")
            self.adapter = nn.Identity()
        elif adapter == "mlp":
            self.adapter = nn.Sequential(
                nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 32)
            )
        elif adapter == "linear":
            self.adapter = nn.Linear(width, 32)
        else:
            raise ValueError(f"unknown adapter: {adapter}")
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, 64, dropout=0, batch_first=True),
            1,
            enable_nested_tensor=False,
        )
        self.head = nn.Linear(32, 29)
        self.representation_shapes = {}

    def forward(self, x, coordinates, subjects, sentences, labels=None):
        features = torch.stack(
            [x.mean(-1), x.square().mean(-1), coordinates[..., 0], coordinates[..., 1]],
            -1,
        )
        weights = torch.softmax(coordinates[..., 0] + coordinates[..., 1], -1)
        local = (self.activation(self.sensor(features)) * weights.unsqueeze(-1)).sum(1)
        local = self.local_norm(local)
        projected = self.adapter(local)
        keys = list(zip(subjects.tolist(), sentences.tolist()))
        groups = [
            [i for i, key in enumerate(keys) if key == identity]
            for identity in dict.fromkeys(keys)
        ]
        length = max(map(len, groups))
        padded = projected.new_zeros(len(groups), length, 32)
        mask = torch.ones(len(groups), length, dtype=torch.bool, device=x.device)
        for group, indices in enumerate(groups):
            padded[group, : len(indices)] = projected[indices]
            mask[group, : len(indices)] = False
        contextual = self.context(padded, src_key_padding_mask=mask)
        ordered = torch.cat(
            [contextual[group, : len(indices)] for group, indices in enumerate(groups)]
        )
        permutation = [i for indices in groups for i in indices]
        contextual = ordered[torch.argsort(torch.tensor(permutation, device=x.device))]
        fused = contextual + projected
        self.representation_shapes = {
            name: list(value.shape)
            for name, value in [
                ("local", local),
                ("contextual", contextual),
                ("fused", fused),
            ]
        }
        return self.head(fused)


MODEL_SPEC = {'width': 48, 'activation': 'gelu', 'adapter': 'linear', 'normalize': False}

class Model(_EmbeddingModel):
    def __init__(self):
        super().__init__(**MODEL_SPEC)
