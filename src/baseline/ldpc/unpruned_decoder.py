"""Work around Sionna 2.0.1's zero-pruning edge case without touching the BR-14 seam.

``LDPC5GDecoder`` prunes punctured degree-1 variable nodes with
``pcm[:-n, :-n]``.  When a code is sent with no parity removed at all — base
graph 2 at its minimum rate 1/5, where ``E = n_ldpc - 2Z`` exactly — ``n`` is 0,
``pcm[:-0, :-0]`` is empty, and every decode fails with a reshape error.  The
unpruned decoder is the same graph with nothing removed, so for that case it is
the identical decoder.  ``adapter.py`` is bound byte-for-byte by closed G-2/G-8
and ER-9 evidence, so this module repairs only adapters that would otherwise be
unusable and leaves every other adapter, and every closed result, unchanged.
"""

from __future__ import annotations

import functools

from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.fec.ldpc.decoding import cn_update_offset_minsum

from config.params import get

from .adapter import SionnaLDPCAdapter


def needs_unpruned_decoder(adapter: SionnaLDPCAdapter) -> bool:
    return int(adapter.decoder.num_vns) == 0


def repair_zero_pruning(adapter: SionnaLDPCAdapter) -> SionnaLDPCAdapter:
    """Rebuild the decoder unpruned only when Sionna produced an empty graph."""

    if not needs_unpruned_decoder(adapter):
        return adapter
    adapter.decoder = LDPC5GDecoder(
        adapter.encoder,
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
        prune_pcm=False,
        device=str(adapter.device),
    ).to(adapter.device)
    if int(adapter.decoder.num_vns) != int(adapter.encoder.n_ldpc):
        raise RuntimeError("unpruned LDPC decoder does not span the full base graph")
    return adapter


__all__ = ["needs_unpruned_decoder", "repair_zero_pruning"]
