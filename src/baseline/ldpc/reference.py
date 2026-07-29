"""Literal batched reconstruction of the pinned independent flooding OMS decoder.

The update order and equations follow the authenticated Lcrypto
``mexFunction/decode.c`` source. This module is gate-only evidence code; the
runtime baseline uses :class:`SionnaLDPCAdapter`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from config.params import get

from .modulation import deinterleave


def load_base_graph(path: Path, lifting_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    shifts = np.loadtxt(path, dtype=np.int64)
    rows, columns = shifts.shape
    edges: list[list[int]] = [[] for _ in range(rows * lifting_size)]
    for block_row in range(rows):
        for block_column in range(columns):
            shift = int(shifts[block_row, block_column])
            if shift < 0:
                continue
            for row in range(lifting_size):
                edges[block_row * lifting_size + row].append(
                    block_column * lifting_size + (row + shift) % lifting_size
                )
    max_degree = max(map(len, edges))
    columns_padded = np.zeros((len(edges), max_degree), dtype=np.int64)
    mask = np.zeros_like(columns_padded, dtype=bool)
    for row, values in enumerate(edges):
        columns_padded[row, : len(values)] = values
        mask[row, : len(values)] = True
    return torch.from_numpy(columns_padded), torch.from_numpy(mask)


class IndependentFloodingOMS:
    """Fixed-iteration flooding offset-min-sum from the pinned source."""

    def __init__(self, graph_path: Path, lifting_size: int, device: str):
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        columns, mask = load_base_graph(graph_path, lifting_size)
        self.columns = columns.to(self.device)
        self.mask = mask.to(self.device)
        self.lifting_size = lifting_size
        self.offset = float(get("baseline.ldpc_decoder_offset"))
        self.llr_clip = float(get("baseline.ldpc_decoder_llr_clip"))
        self.iterations = int(get("baseline.ldpc_max_iters"))
        self.variable_count = int(get("baseline.ldpc_mother_code_columns")["bg2"]) * lifting_size

    def _recover(
        self, llr_log_p1_over_p0: np.ndarray, k: int, q_m: int
    ) -> torch.Tensor:
        values = deinterleave(np.asarray(llr_log_p1_over_p0), q_m)
        received = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        batch, n = received.shape
        z = self.lifting_size
        k_ldpc = 10 * z  # literal-ok: TS 38.212 BG2 systematic columns
        punctured = int(get("baseline.ldpc_punctured_systematic_columns")) * z
        filler = k_ldpc - k
        systematic_received = k - punctured
        if systematic_received < 0 or n < systematic_received:
            raise ValueError("reference rate-recovery dimensions are invalid")
        full = torch.zeros((batch, self.variable_count), dtype=torch.float32, device=self.device)
        full[:, punctured:k] = -received[:, :systematic_received]
        full[:, k:k_ldpc] = self.llr_clip
        parity_count = n - systematic_received
        full[:, k_ldpc:k_ldpc + parity_count] = -received[:, systematic_received:]
        return full.clamp(-self.llr_clip, self.llr_clip)

    def decode(
        self, llr_log_p1_over_p0: np.ndarray, k: int, q_m: int
    ) -> np.ndarray:
        channel = self._recover(llr_log_p1_over_p0, k, q_m)
        batch = channel.shape[0]
        columns = self.columns
        mask = self.mask
        edge_channel = channel[:, columns]
        v2c = edge_channel.clone()
        positive_infinity = torch.finfo(channel.dtype).max
        c2v = torch.zeros_like(v2c)
        for _ in range(self.iterations):
            signed = torch.where(v2c < 0, -torch.ones_like(v2c), torch.ones_like(v2c))
            signed = torch.where(mask, signed, torch.ones_like(signed))
            total_sign = signed.prod(dim=2, keepdim=True)
            magnitudes = torch.where(mask, v2c.abs(), positive_infinity)
            smallest = torch.topk(magnitudes, 2, dim=2, largest=False).values
            minimum, second = smallest[..., :1], smallest[..., 1:2]
            extrinsic = torch.where(magnitudes == minimum, second, minimum)
            c2v = total_sign * signed * torch.clamp(extrinsic - self.offset, min=0)
            c2v = torch.where(mask, c2v, torch.zeros_like(c2v))
            totals = torch.zeros(
                (batch, self.variable_count), dtype=channel.dtype, device=self.device
            )
            totals.scatter_add_(
                1, columns.reshape(1, -1).expand(batch, -1), c2v.reshape(batch, -1)
            )
            v2c = edge_channel + totals[:, columns] - c2v
            v2c = torch.where(mask, v2c, torch.zeros_like(v2c))
        totals = torch.zeros(
            (batch, self.variable_count), dtype=channel.dtype, device=self.device
        )
        totals.scatter_add_(
            1, columns.reshape(1, -1).expand(batch, -1), c2v.reshape(batch, -1)
        )
        hard = (channel + totals < 0).to(torch.uint8)
        return hard[:, :k].cpu().numpy()
