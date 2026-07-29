"""BR-14 substitution seam around Sionna 2.0.1."""

from __future__ import annotations

import functools
import numpy as np
import torch
from sionna import __version__ as sionna_version
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import cn_update_offset_minsum

from config.params import get


class SionnaLDPCAdapter:
    def __init__(self, k: int, n: int, q_m: int, base_graph: int, device: str = "cpu"):
        expected = str(get("baseline.ldpc_impl_version"))
        if sionna_version != expected:
            raise RuntimeError(f"Sionna {sionna_version} != configured {expected}")
        if get("baseline.ldpc_decoder") != "offset_min_sum":
            raise RuntimeError("adapter only implements the configured offset-min-sum decoder")
        if get("baseline.ldpc_decoder_impl_spelling") != "offset-minsum":
            raise RuntimeError("configured Sionna decoder spelling is not recognised")
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        sionna_device = str(self.device)
        self.encoder = LDPC5GEncoder(
            k=k,
            n=n,
            num_bits_per_symbol=q_m,
            bg=f"bg{base_graph}",
            device=sionna_device,
        ).to(self.device)
        self.decoder = LDPC5GDecoder(
            self.encoder,
            cn_update=functools.partial(
                cn_update_offset_minsum,
                offset=float(get("baseline.ldpc_decoder_offset")),
            ),
            vn_update=get("baseline.ldpc_decoder_vn_update"),
            cn_schedule=get("baseline.ldpc_decoder_schedule"),
            hard_out=True,
            return_infobits=True,
            num_iter=int(get("baseline.ldpc_max_iters")),
            llr_max=float(get("baseline.ldpc_decoder_llr_clip")),
            device=sionna_device,
        ).to(self.device)
        if int(self.encoder.z) != self.lifting_size:
            raise AssertionError("encoder lifting-size property is inconsistent")

    @property
    def lifting_size(self) -> int:
        return int(self.encoder.z)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(bits, dtype=torch.float32, device=self.device)
        return self.encoder(tensor).detach().cpu().numpy().astype(np.uint8)

    def decode(self, llr_log_p1_over_p0: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(llr_log_p1_over_p0, dtype=torch.float32, device=self.device)
        return self.decoder(tensor).detach().cpu().numpy().astype(np.uint8)
