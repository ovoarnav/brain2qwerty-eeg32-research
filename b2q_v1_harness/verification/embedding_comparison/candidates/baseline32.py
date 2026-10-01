"""Small synthetic proxy, not the official model or a scientific result."""
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, width=32, fault=None):
        super().__init__()
        self.fault = fault
        self.sensor = nn.Linear(4, width)
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0, batch_first=True),
            1, enable_nested_tensor=False,
        )
        self.head = nn.Linear(width, 29)
        self.representation_shapes = {}

    def forward(self, x, coordinates, subjects, sentences, labels=None):
        # Sensor pooling is invariant to JOINT permutation of signals/coordinates.
        sensor_features = torch.stack(
            [x.mean(-1), x.square().mean(-1), coordinates[..., 0], coordinates[..., 1]], -1
        )
        weights = torch.softmax(coordinates[..., 0] + coordinates[..., 1], -1)
        local = (self.sensor(sensor_features) * weights.unsqueeze(-1)).sum(1)
        if self.fault == 'detached_upstream':
            local = local.detach()
        keys = list(zip(subjects.tolist(), sentences.tolist()))
        if self.fault == 'cross_sentence_mix':
            keys = [(0, 0)] * len(keys)
        unique = list(dict.fromkeys(keys))
        groups = [[i for i, key in enumerate(keys) if key == identity] for identity in unique]
        length = max(map(len, groups))
        padded = local.new_zeros(len(groups), length, local.shape[-1])
        mask = torch.ones(len(groups), length, dtype=torch.bool, device=x.device)
        for g, indices in enumerate(groups):
            padded[g, :len(indices)] = local[indices]
            mask[g, :len(indices)] = False
        contextual = self.context(padded, src_key_padding_mask=None if self.fault == 'padding_mask_removed' else mask)
        ordered = torch.cat([contextual[g, :len(indices)] for g, indices in enumerate(groups)])
        permutation = [i for indices in groups for i in indices]
        inverse = torch.argsort(torch.tensor(permutation, device=x.device))
        contextual = ordered if self.fault == 'row_order_lost' else ordered[inverse]
        fused = contextual + local
        self.representation_shapes = {name: list(value.shape) for name, value in [('local', local), ('contextual', contextual), ('fused', fused)]}
        if self.fault == 'width_mismatch':
            fused = fused[:, :-1]
        logits = self.head(fused)
        if self.fault == 'label_leak' and labels is not None:
            logits = logits + torch.nn.functional.one_hot(labels, 29).to(logits) * 10
        return logits[:-1] if self.fault == 'wrong_output_shape' else logits
